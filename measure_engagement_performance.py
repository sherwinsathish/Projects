import os
import json
import re
import statistics
import random
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
import numpy as np
from inter_annotator_agreement import compute_agreement
import base64

def convert_video_to_base64(video_path):
    with open(video_path, "rb") as video_file:
        video_base64 = base64.b64encode(video_file.read()).decode("utf-8")
    return video_base64

# Example paths for your high and low engagement video files
high_video_path = "/home/mina/Downloads/q1high.mp4"
low_video_path = "/home/mina/Downloads/q1low.mp4"
high_video_pathq4 = "/home/mina/Downloads/q4high.mp4"
q2high = "/home/mina/Downloads/q2high.mp4"
q3high = "/home/mina/Downloads/q3high.mp4"

high_video_base64 = convert_video_to_base64(high_video_path) #q1high,q6high
low_video_base64 = convert_video_to_base64(low_video_path)  #q1low,q2low,q3low,q4low
high_video_base64q4 = convert_video_to_base64(high_video_pathq4) #q4 qnd q5 high, q6low
q2highb64 = convert_video_to_base64(q2high) #q2highnodding
q3highb64 = convert_video_to_base64(q3high) #q3high




def set_seed(seed: int):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)

def get_random_clips(data_dir: str, n: int) -> List[str]:
    """
    Collect ALL .mp4 clips from every folder and subfolder within data_dir,
    and randomly return 'n' clips (or fewer if there are not enough clips).
    """
    all_clips = []
    for session_folder in os.listdir(data_dir):
        session_path = os.path.join(data_dir, session_folder)
        if not os.path.isdir(session_path):
            continue
        for clip_folder in os.listdir(session_path):
            clip_path = os.path.join(session_path, clip_folder)
            if not os.path.isdir(clip_path):
                continue
            for clip_file in os.listdir(clip_path):
                if clip_file.endswith(".mp4"):
                    clip_file_path = os.path.join(clip_path, clip_file)
                    all_clips.append(clip_file_path)
    
    # Randomly sample 'n' clips and print the result
    sampled_clips = random.sample(all_clips, min(n, len(all_clips)))
    print(f"Randomly sampled clips ({len(sampled_clips)}):")
    for clip in sampled_clips:
        print(f"  - {clip}")
    return sampled_clips


def upload_to_gcs(bucket_name: str, file_paths: List[str]) -> List[str]:
    """
    Uploads a list of files to a specified Google Cloud Storage bucket.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    gcs_uris = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_path)
        gcs_uri = f"gs://{bucket_name}/{file_name}"
        gcs_uris.append(gcs_uri)
        print(f"Uploaded {file_name} to {gcs_uri}.")
    return gcs_uris


def get_engagement_scores(csv_dir: str, random_clips: List[str]) -> Dict[str, float]:
    """
    Find engagement scores for the random clips from human-annotated CSV files.
    
    1) We only traverse folders in 'csv_dir' that start with 'label'.
    2) For each clip in 'random_clips', we extract the substring after '/video/'
       and remove the '.mp4' extension to match the 'video_path' column in each CSV.
    3) If there's a match in a CSV, we compute the average engagement score for the clip.
    """
    engagement_scores = {}
    
    # Traverse all subfolders in the da\ta directory
    for folder in os.listdir(csv_dir):
        # Skip folders that don't start with "label"
        if not folder.startswith("label"):
            continue
        
        folder_path = os.path.join(csv_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Look for CSV files
        for csv_file in os.listdir(folder_path):
            if not csv_file.endswith(".csv"):
                continue
            
            csv_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(csv_path)

            # For each randomly sampled clip, parse out the portion after 'video/' and remove '.mp4'
            for clip_path in random_clips:
                # Try to extract the part after "/video/"
                if "/video/" in clip_path:
                    # everything after /video/
                    partial_path = clip_path.split("/video/", 1)[1]
                else:
                    # If there's no "/video/" in the path, skip or continue
                    continue
                
                # Remove the ".mp4" extension
                if partial_path.endswith(".mp4"):
                    partial_path = partial_path[:-4]
                
                # Check matches in the CSV
                matches = df[df["video_path"] == partial_path]
                if not matches.empty:
                    # Initialize if we haven't seen this clip before
                    if clip_path not in engagement_scores:
                        engagement_scores[clip_path] = []
                    # Extend with any found engagement scores
                    engagement_scores[clip_path].extend(matches["engagement"].tolist())
    
    # Compute average engagement scores
    avg_engagement_scores = {}
    for clip, scores in engagement_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0
        avg_engagement_scores[clip] = avg_score
    
    # Print average engagement scores
    print("Average Engagement Scores from human annotators:")
    for clip, avg_score in avg_engagement_scores.items():
        print(f"  - {clip}: {avg_score:.2f}")
    
    return avg_engagement_scores


# Dictionary to store majority votes for each clip
majority_votes_dict = {}

def get_human_annotations_QA(
    random_clips: List[str],
    raw_annot_video_id_path: str,
    raw_annot_dir_path: str,
    merged_df_save_path: str = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieves multiple annotations (Vid_num and Q1-Q6) for each clip by matching
    the portion of the path after '/video/'. If multiple rows have the same partial path,
    they are all stored and labeled as A1, A2, A3, etc.

    Additionally, optionally saves the merged DataFrame to a specified file path.

    :param random_clips: List of full paths to random clips.
    :param raw_annot_video_id_path: File path to the Video ID CSV.
    :param raw_annot_dir_path: File path to the Annotations CSV.
    :param merged_df_save_path: (Optional) File path to save the merged DataFrame as CSV.
    :return: A dictionary { clip_full_path: [ { 'Vid_num': ..., 'Q1': ..., ... }, ... ] }
             where each clip maps to a list of annotation dictionaries.
    """
    majorityindex = 0 
    # Define the mapping for Q1-Q5
    response_mapping_q1_q5 = {
        'Never': 1,
        'Rarely': 2,
        'Sometimes': 3,
        'Often': 4,
        'Very often': 5
    }

    # Define the mapping for Q6
    response_mapping_q6 = {
        'Yes': 5,
        'Unsure': 3,
        'No': 1
    }

    # Read CSV files and drop the first column (assuming it's an unnamed index)
    try:
        video_ids_df = pd.read_csv(raw_annot_video_id_path, sep=',').iloc[:, 1:]
        raw_annotations_df = pd.read_csv(raw_annot_dir_path, sep=',').iloc[:, 1:]
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return {}

    # Check for duplicates in video_ids_df
    duplicate_vid_ids = video_ids_df[video_ids_df.duplicated(subset='Vid_num', keep=False)]
    if not duplicate_vid_ids.empty:
        print("Warning: Duplicate 'Vid_num' found in video_ids_df. These will cause row duplication after merge.")
        print(duplicate_vid_ids)
        # Drop duplicates keeping the first occurrence
        video_ids_df = video_ids_df.drop_duplicates(subset='Vid_num', keep='first')
        print("Duplicates in video_ids_df have been dropped, keeping the first occurrence.")

    # Merge DataFrames on 'Vid_num'
    try:
        merged_df = pd.merge(video_ids_df, raw_annotations_df, on='Vid_num', how='inner')
    except KeyError as e:
        print(f"Merge failed due to missing key: {e}")
        return {}

    # Define Q1-Q6 columns
    q_columns = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6']

    # Check if all Q1-Q6 columns exist
    missing_q_columns = [q for q in q_columns if q not in merged_df.columns]
    if missing_q_columns:
        print(f"Missing Q1-Q6 columns in merged DataFrame: {missing_q_columns}")
        return {}

    # Fill missing Q1-Q6 with empty strings
    merged_df[q_columns] = merged_df[q_columns].fillna('')

    # Remove completely duplicated rows
    merged_df = merged_df.drop_duplicates()

    # Optional: Save the merged DataFrame
    if merged_df_save_path:
        try:
            merged_df.to_csv(merged_df_save_path, index=False)
            print(f"Merged DataFrame saved to {merged_df_save_path}")
        except Exception as e:
            print(f"Error saving merged DataFrame: {e}")

    # Initialize the dictionary to hold video paths -> list of annotation dictionaries
    video_to_questions: Dict[str, List[Dict[str, Any]]] = {}

    for _, row in merged_df.iterrows():
        # Convert 'Video Link' to string to handle NaN or other non-string types
        video_link = str(row.get("Video Link", "nan")).strip()  # Safely get 'Video Link'

        # Check if 'Video Link' is 'nan' or empty after stripping
        if video_link.lower() == "nan" or video_link == "":
            continue  # Skip rows with missing/invalid link

        # Extract path after "/video/" and remove ".mp4"
        if "/video/" in video_link:
            partial_path = video_link.split("/video/", 1)[1]
        else:
            partial_path = video_link  # Fallback if "/video/" not found

        if partial_path.endswith(".mp4"):
            partial_path = partial_path[:-4]

        # Normalize the path and convert to lowercase for consistency
        normalized_partial = os.path.normpath(partial_path).lower()

        # Map Q1-Q5 responses to numeric values
        mapped_q1_q5 = {}
        for q in q_columns[:5]:  # Q1 to Q5
            response = row[q].strip()
            mapped_value = response_mapping_q1_q5.get(response, None)  # Assign None if response not found
            if mapped_value is None and response != '':
                print(f"Warning: Unmapped response '{response}' for {q} in Vid_num {row['Vid_num']}. Assigning None.")
            mapped_q1_q5[q] = mapped_value

        # Map Q6 response to numeric value
        response_q6 = row['Q6'].strip()
        if response_q6 != '':
            mapped_q6 = response_mapping_q6.get(response_q6, None)
            if mapped_q6 is None and response_q6 not in response_mapping_q6:
                print(f"Warning: Unmapped response '{response_q6}' for Q6 in Vid_num {row['Vid_num']}. Assigning None.")
        else:
            mapped_q6 = None
        mapped_q1_q5['Q6'] = mapped_q6

        # Build a dictionary of annotation values for this row with numeric responses
        row_annotations = {
            'Vid_num': row['Vid_num'],
            **mapped_q1_q5
        }

        # Append to the list for this partial path
        if normalized_partial not in video_to_questions:
            video_to_questions[normalized_partial] = []
        video_to_questions[normalized_partial].append(row_annotations)

    # Build the result dictionary by matching random_clips to the list of annotations
    result: Dict[str, List[Dict[str, Any]]] = {}
    for clip in random_clips:
        # Extract partial path from the clip's full path
        if "/video/" in clip:
            clip_partial = clip.split("/video/", 1)[1]
        else:
            clip_partial = clip  # Fallback

        if clip_partial.endswith(".mp4"):
            clip_partial = clip_partial[:-4]

        # Normalize the partial path and convert to lowercase for consistency
        normalized_clip_partial = os.path.normpath(clip_partial).lower()

        # Get all Vid_num and Q1-Q6 matches for this clip
        # or default to an empty list if no matches
        annotation_list = video_to_questions.get(normalized_clip_partial, [])

        # Save them under the clip's full path
        result[clip] = annotation_list

    # Print results
    print("\nVid_num and Q1-Q6 (multiple annotators) for each clip:")
    for clip, annotations in result.items():
        print(f"Clip: {clip}")
        if not annotations:
            print("  (No matching annotations found)\n")
            continue

        # Dictionary to count "Y" and "N" votes for each question
        majority_count = {q: {"Y": 0, "N": 0} for q in q_columns}

        # Print each annotation, labeled A1, A2, etc.
        for i, data in enumerate(annotations, start=1):
            print(f"  Annotator A{i}:")
            print(f"    Vid_num: {data.get('Vid_num', '')}")
            for q in q_columns:
                value = data.get(q, '')
                if isinstance(value, (int, float)):  # Ensure it's a number before mapping
                    yn_value = "Y" if value in [4, 5] else "N"
                    majority_count[q][yn_value] += 1  # Update majority count
                else:
                    yn_value = value  # Keep as is if not a number
                print(f"    {q}: {yn_value}")
            print()

        # Compute majority vote for each question
        majority_vote = {}
        print(f"  Majority Annotator Vote for the clip: {clip}")
        for q in q_columns:
            if majority_count[q]["Y"] > majority_count[q]["N"]:
                majority_vote[q] = "Y"
            elif majority_count[q]["Y"] < majority_count[q]["N"]:
                majority_vote[q] = "N"
            else:
                majority_vote[q] = "T"  # Tie case

            print(f"    {q}: {majority_vote[q]}")
        print()
        majority_votes_dict[majorityindex] = majority_vote
        majorityindex += 1
    return result



def compute_overall_engagement(annotation: Dict[str, int]) -> float:
    """
    Computes the overall engagement score for a single annotation.
    
    :param annotation: Dictionary with keys 'Q1' to 'Q6', each an integer in [1..5].
    :return: Overall engagement score as a float in [-1, 1].
    """
    try:
        a1 = annotation["Q1"]
        a2 = annotation["Q2"]
        a3 = annotation["Q3"]
        a4 = annotation["Q4"]
        a5 = annotation["Q5"]
        a6 = annotation["Q6"]
        e = (a1 + a2 + a3 - a4 - a5 + a6 - 6) / 12.0
        return e
    except KeyError as e:
        print(f"Missing key in annotation: {e}")
        return None
    except TypeError as e:
        print(f"Invalid type in annotation: {e}")
        return None


def aggregate_human_engagement_scores(
    human_annotations: List[Dict[str, Any]], 
    method: str = "mean"
) -> Optional[float]:
    """
    Aggregates multiple human engagement scores into a single score per clip.
    
    :param human_annotations: List of annotation dictionaries for a single clip.
    :param method: Aggregation method - 'mean' or 'median'.
    :return: Aggregated engagement score as a float in [-1, 1], or None if no valid annotations.
    """
    if not human_annotations:
        return None
    
    # Compute 'e' for each annotator
    scores = []
    for ann in human_annotations:
        if all(q in ann and isinstance(ann[q], (int, float)) for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]):
            e = compute_overall_engagement(ann)
            if e is not None:
                scores.append(e)
        else:
            print(f"Invalid annotation detected and skipped: {ann}")
    
    if not scores:
        return None
    
    if method == "mean":
        return sum(scores) / len(scores)
    elif method == "median":
        return statistics.median(scores)
    else:
        raise ValueError(f"Unsupported aggregation method: {method}")

def report_human_llm_engagement_scores(
    human_data: Dict[str, List[Dict[str, Any]]],
    llm_data: Dict[str, Dict[str, Any]],
    agg_method: str = "mean"
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Builds dictionaries of human and LLM engagement scores per clip and prints them.
    
    :param human_data: Dictionary mapping clip full paths to lists of human annotation dicts.
    :param llm_data: Dictionary mapping clip full paths to single LLM annotation dicts.
    :param agg_method: Aggregation method for human annotations - 'mean' or 'median'.
    :return: Tuple of two dictionaries:
             - human_scores: {clip_full_path: human_engagement_score}
             - llm_scores: {clip_full_path: llm_engagement_score}
    """
    human_scores = {}
    llm_scores = {}
    
    print("\n--- Human and LLM Engagement Scores ---\n")
    
    for clip_path, ann_list in human_data.items():
        # Check if LLM has annotated this clip
        if clip_path not in llm_data:
            print(f"LLM annotation missing for clip: {clip_path}")
            continue
        
        # Aggregate human scores
        e_human = aggregate_human_engagement_scores(ann_list, method=agg_method)
        if e_human is None:
            print(f"No valid human annotations for clip: {clip_path}")
            continue
        
        # Compute LLM's engagement score
        llm_ann = llm_data[clip_path]
        if not all(q in llm_ann for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]):
            print(f"LLM annotation incomplete for clip: {clip_path}")
            continue
        
        e_llm = compute_overall_engagement(llm_ann)
        if e_llm is None:
            print(f"Failed to compute LLM engagement score for clip: {clip_path}")
            continue
        
        # Store the scores
        human_scores[clip_path] = e_human
        llm_scores[clip_path] = e_llm
        
        # Print the scores
        print(f"Clip: {clip_path}")
        print(f"  Human Engagement Score: {e_human:.3f}")
        print(f"  LLM Engagement Score: {e_llm:.3f}\n")
    
    return human_scores, llm_scores


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """
    Parses the LLM response text and extracts Q1-Q6 values.

    :param response_text: LLM-generated response as a JSON string.
    :return: Dictionary with Q1-Q6 values as integers, or None if parsing fails.
    """
    try:
        # Remove extra whitespace and attempt to parse JSON
        response_text = response_text.strip()
        llm_data = json.loads(response_text)

        # Ensure all required questions are present
        required_keys = {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}
        if not required_keys.issubset(llm_data.keys()):
            print(f"Missing required keys in response: {response_text}")
            return None

        # Convert Q6 if it's in text format (Yes/Unsure/No -> 5/3/1)
        q6_mapping = {"Yes": 5, "Unsure": 3, "No": 1}
        if isinstance(llm_data["Q6"], str):
            llm_data["Q6"] = q6_mapping.get(llm_data["Q6"], None)

        # Ensure all Q1-Q6 values are integers in the valid range [1, 5]
        for q in required_keys:
            if not isinstance(llm_data[q], int) or llm_data[q] not in range(1, 6):
                print(f"Invalid value for {q}: {llm_data[q]}")
                return None

        return llm_data  # Valid parsed LLM response

    except json.JSONDecodeError:
        print(f"Failed to parse LLM response as JSON: {response_text}")
        return None
    


# Dictionary to store majority votes for each clip
llm_votes_dict = {}

def analyze_clips_with_gemini(
        project_id: str, 
        location: str, 
        bucket_name: str, 
        clip_names: List[str],
        clip_full_paths: List[str]
    ) -> Dict[str, Dict[str, Any]]:
    """
    Analyzes video clips using the Gemini model in Vertex AI and returns LLM annotations.
    
    :param project_id: Google Cloud project ID.
    :param location: Google Cloud location.
    :param bucket_name: Name of the GCS bucket where clips are stored.
    :param clip_names: List of clip filenames to analyze.
    :param clip_full_paths: List of full paths corresponding to clip_names.
    :return: Dictionary mapping full clip paths to LLM annotation dicts.
    """
    vertexai.init(project=project_id, location=location)
    model = GenerativeModel("gemini-1.5-flash-001")
    generation_config = GenerationConfig(temperature=0.5)
    llm_annotations = {}
    llmdictindex=0

    for clip_name, clip_full_path in zip(clip_names, clip_full_paths):
        gcs_uri = f"gs://{bucket_name}/{clip_name}"
        video_part = Part.from_uri(gcs_uri, mime_type="video/mp4")

        contents = [
            video_part,
            "Task: Analyze the level of listener engagement in the video clip.",
            "Context: You are an annotator evaluating video clips for engagement behaviors.",
            "Your goal is to provide a fair, unbiased rating based on visible listener behaviors.",
            "Annotation Guidelines:",
            "- Each video clip is 5–10 seconds long and features a listener.",
            "- Ensure you have a clear view of the listener's reactions.",
            "- Skip clips if video quality is poor, the listener is speaking, or no one is visible.",
            "Rating Scale:",
            "- Questions Q1-Q5: Rate on a scale of 1 (Never) to 5 (Very Often).",
            "- Question Q6: Choose one of the three options — 5 for Yes, 3 for Unsure, or 1 for No.",
            "Examples for Individual Questions:",
            "Q1: 'Did the listener produce verbal/audible sounds in response to the conversation?'",
            "- High Score (5): The listener frequently says 'uh-huh,' 'yeah,' or other verbal affirmations.",
            "- Low Score (1): The listener remains completely silent throughout the clip.",
            "Q2: 'Did the listener nod or shake their head in response to the conversation?'",
            "- High Score (5): The listener nods multiple times, indicating active engagement.",
            "- Low Score (1): The listener never nods or moves their head in response.",
            "Q3: 'Did the listener make facial expressions in response to the conversation?'",
            "- High Score (5): The listener's face visibly changes (e.g., smiles, raises eyebrows) multiple times.",
            "- Low Score (1): The listener maintains a blank or neutral expression the entire time.",
            "Q4: 'Was the listener engaged in another activity while the speaker was presenting?'",
            "- High Score (5): The listener is completely distracted, looking at their phone or writing something.",
            "- Low Score (1): The listener is fully focused on the speaker and not doing anything else.",
            "Q5: 'Did the listener look away from the speaker?'",
            "- High Score (5): The listener frequently looks away and does not maintain eye contact.",
            "- Low Score (1): The listener maintains eye contact with the speaker the entire time.",          
            "Q6: 'Was the listener socially engaged with the speaker?' Answer this question with a Yes, Unsure, or No",
            "- Yes example: The listener is visibly engaged, responding with gestures, expressions, or nods.",
            "- No example: The listener appears uninterested, giving no visible reactions.",        
            "Your Task: Analyze the provided video and answer the following questions.",
            "Answer Format: Provide answers in a JSON object with keys 'Q1' to 'Q5' and numeric values. For Q6 put it as Yes, Unsure, or No",
            "Example Response Format: {\"Q1\": 2, \"Q2\": 4, \"Q3\": 3, \"Q4\": 1, \"Q5\": 2, \"Q6\": \"No\"}"
        ]

        # Get response from LLM
        response = model.generate_content(contents=contents, generation_config=generation_config)
        print(f"\nResults for video: {gcs_uri}")
        print(response.text)

        # Directly parse the LLM response
        llm_annotation = parse_llm_response(response.text)

        # Convert Q6 response if it's in text format
        if llm_annotation and isinstance(llm_annotation.get("Q6"), str):
            q6_mapping = {"Yes": 5, "Unsure": 3, "No": 1}
            llm_annotation["Q6"] = q6_mapping.get(llm_annotation["Q6"], None)

        # Validate and store only complete annotations
        if llm_annotation and all(isinstance(llm_annotation.get(q), (int, float)) for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]):
            llm_annotations[clip_full_path] = llm_annotation

            # Convert numeric scores to Y/N representation
            yn_annotation = {}
            print("\n--- LLM Gemini Annotator Scores ---")
            for q in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
                value = llm_annotation.get(q, '')
                yn_value = "Y" if value in [4, 5] else "N" if isinstance(value, (int, float)) else value
                yn_annotation[q] = yn_value
                print(f"  {q}: {yn_value}")  # Print Y/N format

        else:
            print(f"Skipping incomplete or invalid LLM annotation for {clip_full_path}.")

        llm_votes_dict[llmdictindex] = yn_annotation
        llmdictindex += 1

    return llm_annotations





if __name__ == "__main__":
    SEED = 99
    DATA_DIR = "/home/mina/Downloads/SEMPI_private/data/engagement/video"
    CSV_DIR = "/home/mina/Downloads/SEMPI_public/data/engagement"
    merged_df_save_path = 'merged_dataframe.csv'  
    RAW_ANNOT_DIR = "/home/mina/Desktop/yashcode/raw_annotations.csv"
    RAW_ANNOT_VIDEO_ID = "/home/mina/Desktop/yashcode/raw_annotations_video_ids.csv"
    N_CLIPS = 5
    BUCKET_NAME = "hcc-bucket"
    PROJECT_ID = "beaming-age-448621-p9"
    LOCATION = "us-central1"

    set_seed(SEED)
    
    # Set Google Cloud credentials
    """credential_path = "/Users/yashsahitya/Documents/annotatevideos-3e980a57059b.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path"""
    
    # Get random clips
    random_clips = get_random_clips(DATA_DIR, N_CLIPS)
    
    # Extract clip names and full paths
    clip_names = [os.path.basename(clip) for clip in random_clips]
    clip_full_paths = random_clips  # Full paths

    # Upload clips to GCS
    gcs_uris = upload_to_gcs(BUCKET_NAME, random_clips)
    
    # Get human annotations
    human_annotations = get_human_annotations_QA(
        random_clips, 
        RAW_ANNOT_VIDEO_ID, 
        RAW_ANNOT_DIR,
        merged_df_save_path
    )
    
    # Analyze clips with Gemini (LLM) and get LLM annotations
    llm_annotations = analyze_clips_with_gemini(
        PROJECT_ID, 
        LOCATION, 
        BUCKET_NAME, 
        clip_names, 
        clip_full_paths,
    )
    
    # **Compute Inter-Annotator Agreement BEFORE Aggregating Scores**
    agreement_results = {}
    """
    print("\n--- Inter-Annotator Agreement Scores (Fleiss' Kappa & Krippendorff's Alpha) ---\n")

    for question in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]:
        agreement_scores = compute_agreement(human_annotations, question, llm_annotations)
      
        agreement_results[question] = agreement_scores
        
        print(f"{question}:")
        print(f"  Fleiss' Kappa: {agreement_scores['fleiss_kappa']:.3f}")
        print(f"  Krippendorff's Alpha: {agreement_scores['krippendorff_alpha']:.3f}\n")
    
    # Optional: Store agreement scores for later analysis
    with open("agreement_results.json", "w") as f:
        json.dump(agreement_results, f, indent=4)

    print("Agreement scores saved in agreement_results.json")"""

    # Aggregate Human and LLM Engagement Scores
    human_scores, llm_scores = report_human_llm_engagement_scores(
        human_annotations, 
        llm_annotations, 
        agg_method="mean"  # or "median"
    )

def calculate_icr(llm_votes_dict: Dict[int, Dict[str, str]], 
                  majority_votes_dict: Dict[int, Dict[str, str]]) -> Dict[int, Dict[str, float]]:
    """
    Calculates and prints Inter-Coder Reliability (ICR) using percent agreement per clip per question.

    :param llm_votes_dict: Dictionary of LLM annotations {clip_index: {Q1: Y/N, Q2: Y/N, ...}}.
    :param majority_votes_dict: Dictionary of majority annotations {clip_index: {Q1: Y/N, Q2: Y/N, ...}}.
    :return: Dictionary with ICR scores {clip_index: {Q1: agreement%, Q2: agreement%, ...}}.
    """
    icr_scores = {}

    print("\n=== Inter-Coder Reliability (ICR) Scores ===\n")
    
    for clip_index in llm_votes_dict.keys():
        if clip_index not in majority_votes_dict:
            continue  # Skip if there's no majority annotation for this clip

        icr_scores[clip_index] = {}  # Initialize ICR scores for this clip
        llm_answers = llm_votes_dict[clip_index]
        majority_answers = majority_votes_dict[clip_index]

        print(f"Clip {clip_index}:")
        agreements = 0  # Count agreements for overall clip score
        total_questions = len(llm_answers)

        for question in llm_answers.keys():
            llm_value = llm_answers.get(question)
            majority_value = majority_answers.get(question)

            # Ensure both have valid values before comparing
            if llm_value is not None and majority_value is not None:
                agreement = 1.0 if llm_value == majority_value else 0.0
                icr_scores[clip_index][question] = agreement  # Store agreement score (1 or 0)
                agreements += agreement

                # Print comparison results
                match_status = "Match" if agreement == 1.0 else "Mismatch"
                print(f"  {question}: LLM = {llm_value}, Majority = {majority_value} → {match_status}")

        # Compute and print overall agreement percentage for the clip
        overall_agreement = (agreements / total_questions) * 100
        print(f"  Overall Agreement: {overall_agreement:.2f}%\n")

    return ""

calculate_icr(llm_votes_dict,majority_votes_dict)
