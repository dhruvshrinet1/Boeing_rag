"""
Evaluation Script for Boeing 737 Manual RAG Service
Runs all questions from the evaluation Excel file and generates results.
Includes retrieval scoring with Precision, Recall, and F1 metrics.
"""
import pandas as pd
import requests
import time
from datetime import datetime
import json
import re
from typing import List, Set, Tuple


def parse_page_numbers(page_ref) -> Set[int]:
    """
    Parse page reference into a set of page numbers.
    Handles formats like: 83, "83, 104", "41-43", etc.

    Args:
        page_ref: Page reference (int, str, or float)

    Returns:
        Set of page numbers
    """
    pages = set()

    if pd.isna(page_ref):
        return pages

    # Convert to string
    page_str = str(page_ref).strip()

    # Remove any whitespace and split by comma
    parts = [p.strip() for p in page_str.split(',')]

    for part in parts:
        # Check for range (e.g., "41-43")
        if '-' in part:
            try:
                start, end = part.split('-')
                start_num = int(start.strip())
                end_num = int(end.strip())
                pages.update(range(start_num, end_num + 1))
            except ValueError:
                pass
        else:
            # Single page number
            try:
                pages.add(int(float(part)))
            except ValueError:
                pass

    return pages


def calculate_retrieval_metrics(expected_pages: Set[int], retrieved_pages: List[int]) -> Tuple[float, float, float]:
    """
    Calculate Precision, Recall, and F1 score for retrieval.

    Args:
        expected_pages: Set of ground truth page numbers
        retrieved_pages: List of retrieved page numbers

    Returns:
        Tuple of (precision, recall, f1_score)
    """
    if not expected_pages and not retrieved_pages:
        # No pages expected and none retrieved - perfect match
        return 1.0, 1.0, 1.0

    if not expected_pages or not retrieved_pages:
        # One is empty but not the other - zero score
        return 0.0, 0.0, 0.0

    retrieved_set = set(retrieved_pages)

    # Calculate metrics
    true_positives = len(expected_pages & retrieved_set)

    # Precision: Of all pages retrieved, how many were relevant?
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0

    # Recall: Of all relevant pages, how many were retrieved?
    recall = true_positives / len(expected_pages) if expected_pages else 0.0

    # F1: Harmonic mean of precision and recall
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return precision, recall, f1_score


def run_evaluation(
    excel_path: str = "data/document_analysis/Coding Challenge Evals.xlsx",
    api_url: str = "http://localhost:8000/query",
    output_path: str = "evaluation_results.xlsx"
):
    """
    Run evaluation on all questions from the Excel file.

    Args:
        excel_path: Path to the evaluation Excel file
        api_url: URL of the RAG API query endpoint
        output_path: Path to save results Excel file
    """
    print("=" * 80)
    print("Boeing 737 Manual RAG Service - Evaluation with Retrieval Scoring")
    print("=" * 80)
    print()

    # Read evaluation questions
    print(f"Reading evaluation file: {excel_path}")
    df = pd.read_excel(excel_path)
    print(f"Found {len(df)} questions to evaluate")
    print()

    # Check if API is running
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ ERROR: API is not healthy. Please start the server with: python main.py")
            return
        print("✓ API server is running")
    except Exception as e:
        print(f"❌ ERROR: Cannot connect to API server at http://localhost:8000")
        print(f"   Please start the server with: python main.py")
        print(f"   Error: {e}")
        return

    print()
    print("Starting evaluation...")
    print("-" * 80)

    # Store results
    results = []

    # Process each question
    for idx, row in df.iterrows():
        question_num = idx + 1
        question = row['Question']
        expected_answer = row['Labelled Answer']
        expected_pages_raw = row['Page reference']

        # Parse expected pages
        expected_pages = parse_page_numbers(expected_pages_raw)

        print(f"\n[Question {question_num}/{len(df)}]")
        print(f"Q: {question[:100]}...")

        try:
            # Call API
            start_time = time.time()
            response = requests.post(
                api_url,
                json={"question": question},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                actual_answer = result['answer']
                actual_pages = result['pages']

                # Calculate retrieval metrics
                precision, recall, f1 = calculate_retrieval_metrics(expected_pages, actual_pages)

                print(f"✓ Success ({elapsed_time:.2f}s)")
                print(f"  Expected pages: {sorted(expected_pages)}")
                print(f"  Retrieved pages: {actual_pages}")
                print(f"  Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}")

                # Store results
                results.append({
                    'Question_Number': question_num,
                    'Question': question,
                    'Expected_Answer': expected_answer,
                    'Expected_Pages': str(expected_pages_raw),
                    'Expected_Pages_Parsed': sorted(list(expected_pages)),
                    'API_Answer': actual_answer,
                    'API_Pages': str(actual_pages),
                    'API_Pages_List': actual_pages,
                    'Precision': round(precision, 4),
                    'Recall': round(recall, 4),
                    'F1_Score': round(f1, 4),
                    'Retrieval_Score': round(f1, 4),  # Primary metric
                    'Response_Time_Seconds': round(elapsed_time, 2),
                    'Status': 'Success'
                })

            else:
                print(f"✗ API Error: Status {response.status_code}")
                print(f"  Response: {response.text[:200]}")

                results.append({
                    'Question_Number': question_num,
                    'Question': question,
                    'Expected_Answer': expected_answer,
                    'Expected_Pages': str(expected_pages_raw),
                    'Expected_Pages_Parsed': sorted(list(expected_pages)),
                    'API_Answer': f"ERROR: {response.text[:100]}",
                    'API_Pages': 'N/A',
                    'API_Pages_List': [],
                    'Precision': 0.0,
                    'Recall': 0.0,
                    'F1_Score': 0.0,
                    'Retrieval_Score': 0.0,
                    'Response_Time_Seconds': round(elapsed_time, 2),
                    'Status': f'Error_{response.status_code}'
                })

        except Exception as e:
            print(f"✗ Exception: {str(e)}")

            results.append({
                'Question_Number': question_num,
                'Question': question,
                'Expected_Answer': expected_answer,
                'Expected_Pages': str(expected_pages_raw),
                'Expected_Pages_Parsed': sorted(list(expected_pages)),
                'API_Answer': f"ERROR: {str(e)}",
                'API_Pages': 'N/A',
                'API_Pages_List': [],
                'Precision': 0.0,
                'Recall': 0.0,
                'F1_Score': 0.0,
                'Retrieval_Score': 0.0,
                'Response_Time_Seconds': 0,
                'Status': 'Exception'
            })

        # Small delay between requests
        time.sleep(0.5)

    print()
    print("-" * 80)
    print("Evaluation complete!")
    print()

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Calculate overall metrics (only for successful responses)
    successful_results = [r for r in results if r['Status'] == 'Success']

    if successful_results:
        avg_precision = sum(r['Precision'] for r in successful_results) / len(successful_results)
        avg_recall = sum(r['Recall'] for r in successful_results) / len(successful_results)
        avg_f1 = sum(r['F1_Score'] for r in successful_results) / len(successful_results)
        avg_retrieval_score = avg_f1  # Primary metric
    else:
        avg_precision = avg_recall = avg_f1 = avg_retrieval_score = 0.0

    # Save to Excel with multiple sheets
    print(f"Saving results to: {output_path}")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Main results
        results_df.to_excel(writer, sheet_name='Results', index=False)

        # Summary statistics
        summary_data = {
            'Metric': [
                'Total Questions',
                'Successful Responses',
                'Failed Responses',
                '--- RETRIEVAL METRICS ---',
                'Average Retrieval Score (F1)',
                'Average Precision',
                'Average Recall',
                'Perfect Retrievals (F1=1.0)',
                'Failed Retrievals (F1=0.0)',
                '--- PERFORMANCE ---',
                'Average Response Time (s)',
                'Evaluation Date',
                'API URL'
            ],
            'Value': [
                len(results),
                len(successful_results),
                len(results) - len(successful_results),
                '',
                f'{avg_retrieval_score:.2%}',
                f'{avg_precision:.2%}',
                f'{avg_recall:.2%}',
                sum(1 for r in successful_results if r['F1_Score'] == 1.0),
                sum(1 for r in successful_results if r['F1_Score'] == 0.0),
                '',
                round(sum(r['Response_Time_Seconds'] for r in results) / len(results), 2) if results else 0,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                api_url
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

        # Retrieval metrics detail
        if successful_results:
            metrics_df = results_df[results_df['Status'] == 'Success'][
                ['Question_Number', 'Expected_Pages_Parsed', 'API_Pages_List',
                 'Precision', 'Recall', 'F1_Score']
            ].copy()
            metrics_df = metrics_df.sort_values('F1_Score', ascending=False)
            metrics_df.to_excel(writer, sheet_name='Retrieval_Metrics', index=False)

    print(f"✓ Results saved to {output_path}")
    print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Questions:        {len(results)}")
    print(f"Successful Responses:   {len(successful_results)}")
    print(f"Failed Responses:       {len(results) - len(successful_results)}")
    print()
    print("RETRIEVAL METRICS:")
    print(f"  Average Retrieval Score (F1): {avg_retrieval_score:.2%}")
    print(f"  Average Precision:            {avg_precision:.2%}")
    print(f"  Average Recall:               {avg_recall:.2%}")
    print(f"  Perfect Retrievals:           {sum(1 for r in successful_results if r['F1_Score'] == 1.0)}/{len(successful_results)}")
    print(f"  Failed Retrievals:            {sum(1 for r in successful_results if r['F1_Score'] == 0.0)}/{len(successful_results)}")
    print()
    print(f"Average Response Time:  {sum(r['Response_Time_Seconds'] for r in results) / len(results):.2f}s")
    print("=" * 80)
    print()

    # Show sample results
    if successful_results:
        print("Sample Results (First Question):")
        print("-" * 80)
        first = successful_results[0]
        print(f"Question: {first['Question'][:100]}...")
        print(f"\nExpected Pages:  {first['Expected_Pages_Parsed']}")
        print(f"Retrieved Pages: {first['API_Pages_List']}")
        print(f"\nRetrieval Metrics:")
        print(f"  Precision: {first['Precision']:.2%}")
        print(f"  Recall:    {first['Recall']:.2%}")
        print(f"  F1 Score:  {first['F1_Score']:.2%}")
        print(f"\nAnswer Preview: {first['API_Answer'][:200]}...")
        print("-" * 80)

        # Show best and worst retrievals
        if len(successful_results) > 1:
            print()
            best = max(successful_results, key=lambda x: x['F1_Score'])
            worst = min(successful_results, key=lambda x: x['F1_Score'])

            print(f"Best Retrieval (F1={best['F1_Score']:.2%}):")
            print(f"  Q{best['Question_Number']}: {best['Question'][:80]}...")
            print(f"  Expected: {best['Expected_Pages_Parsed']} | Got: {best['API_Pages_List']}")
            print()
            print(f"Worst Retrieval (F1={worst['F1_Score']:.2%}):")
            print(f"  Q{worst['Question_Number']}: {worst['Question'][:80]}...")
            print(f"  Expected: {worst['Expected_Pages_Parsed']} | Got: {worst['API_Pages_List']}")
            print("-" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Boeing RAG evaluation with retrieval scoring")
    parser.add_argument(
        "--excel",
        default="data/document_analysis/Coding Challenge Evals.xlsx",
        help="Path to evaluation Excel file"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000/query",
        help="API query endpoint URL"
    )
    parser.add_argument(
        "--output",
        default="evaluation_results.xlsx",
        help="Output Excel file path"
    )

    args = parser.parse_args()

    run_evaluation(
        excel_path=args.excel,
        api_url=args.api_url,
        output_path=args.output
    )
