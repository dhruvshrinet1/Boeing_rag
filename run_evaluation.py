"""
Evaluation script for Boeing 737 Manual RAG Service
Calculates Precision, Recall, and F1 scores for retrieval
"""
import pandas as pd
import requests
import time
from datetime import datetime
import re
from typing import List, Set, Tuple


def parse_page_numbers(page_ref) -> Set[int]:
    """Parse page reference into set of page numbers (handles ranges like "41-43")"""
    pages = set()

    if pd.isna(page_ref):
        return pages

    page_str = str(page_ref).strip()
    parts = [p.strip() for p in page_str.split(',')]

    for part in parts:
        if '-' in part:
            try:
                start, end = part.split('-')
                pages.update(range(int(start.strip()), int(end.strip()) + 1))
            except ValueError:
                pass
        else:
            try:
                pages.add(int(float(part)))
            except ValueError:
                pass

    return pages


def calculate_retrieval_metrics(expected_pages: Set[int], retrieved_pages: List[int]) -> Tuple[float, float, float]:
    """Calculate Precision, Recall, and F1 score"""
    if not expected_pages and not retrieved_pages:
        return 1.0, 1.0, 1.0

    if not expected_pages or not retrieved_pages:
        return 0.0, 0.0, 0.0

    retrieved_set = set(retrieved_pages)
    true_positives = len(expected_pages & retrieved_set)

    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(expected_pages) if expected_pages else 0.0

    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    return precision, recall, f1_score


def run_evaluation(
    excel_path: str = "data/document_analysis/Coding Challenge Evals.xlsx",
    api_url: str = "http://localhost:8080/query",
    output_path: str = "evaluation_results.xlsx"
):
    print("=" * 70)
    print("Boeing 737 Manual RAG - Evaluation with Retrieval Scoring")
    print("=" * 70)

    # load questions
    df = pd.read_excel(excel_path)
    print(f"\nLoaded {len(df)} questions from {excel_path}")

    # check API
    try:
        health = requests.get("http://localhost:8000/health", timeout=5)
        if health.status_code != 200:
            print("❌ API not running. Start with: python main.py")
            return
        print("✓ API is up")
    except Exception as e:
        print(f"❌ Can't connect to API: {e}")
        return

    print("\nRunning evaluation...")
    print("-" * 70)

    results = []

    for idx, row in df.iterrows():
        question_num = idx + 1
        question = row['Question']
        expected_answer = row['Labelled Answer']
        expected_pages_raw = row['Page reference']
        expected_pages = parse_page_numbers(expected_pages_raw)

        print(f"\n[{question_num}/{len(df)}] {question[:80]}...")

        try:
            start_time = time.time()
            response = requests.post(
                api_url,
                json={"question": question},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                actual_answer = result['answer']
                actual_pages = result['pages']

                precision, recall, f1 = calculate_retrieval_metrics(expected_pages, actual_pages)

                print(f"  ✓ {elapsed:.1f}s | P:{precision:.2%} R:{recall:.2%} F1:{f1:.2%}")
                print(f"  Expected: {sorted(expected_pages)} | Got: {actual_pages}")

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
                    'Response_Time_Seconds': round(elapsed, 2),
                    'Status': 'Success'
                })
            else:
                print(f"  ✗ API Error: {response.status_code}")
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
                    'Response_Time_Seconds': round(elapsed, 2),
                    'Status': f'Error_{response.status_code}'
                })

        except Exception as e:
            print(f"  ✗ Exception: {str(e)}")
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
                'Response_Time_Seconds': 0,
                'Status': 'Exception'
            })

        time.sleep(0.5)

    print("\n" + "-" * 70)
    print("Done!\n")

    # save results
    results_df = pd.DataFrame(results)
    successful_results = [r for r in results if r['Status'] == 'Success']

    if successful_results:
        avg_precision = sum(r['Precision'] for r in successful_results) / len(successful_results)
        avg_recall = sum(r['Recall'] for r in successful_results) / len(successful_results)
        avg_f1 = sum(r['F1_Score'] for r in successful_results) / len(successful_results)
    else:
        avg_precision = avg_recall = avg_f1 = 0.0

    # write Excel with multiple sheets
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='Results', index=False)

        # summary stats
        summary_data = {
            'Metric': [
                'Total Questions',
                'Successful Responses',
                'Failed Responses',
                '',
                'Average F1 Score',
                'Average Precision',
                'Average Recall',
                'Perfect Retrievals (F1=1.0)',
                'Zero Retrievals (F1=0.0)',
                '',
                'Avg Response Time (s)',
                'Evaluation Date'
            ],
            'Value': [
                len(results),
                len(successful_results),
                len(results) - len(successful_results),
                '',
                f'{avg_f1:.2%}',
                f'{avg_precision:.2%}',
                f'{avg_recall:.2%}',
                sum(1 for r in successful_results if r['F1_Score'] == 1.0),
                sum(1 for r in successful_results if r['F1_Score'] == 0.0),
                '',
                round(sum(r['Response_Time_Seconds'] for r in results) / len(results), 2) if results else 0,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

        # retrieval metrics detail
        if successful_results:
            metrics_df = results_df[results_df['Status'] == 'Success'][
                ['Question_Number', 'Expected_Pages_Parsed', 'API_Pages_List',
                 'Precision', 'Recall', 'F1_Score']
            ].copy()
            metrics_df = metrics_df.sort_values('F1_Score', ascending=False)
            metrics_df.to_excel(writer, sheet_name='Retrieval_Metrics', index=False)

    print(f"Results saved to: {output_path}\n")

    # print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total Questions:        {len(results)}")
    print(f"Successful:             {len(successful_results)}")
    print(f"Failed:                 {len(results) - len(successful_results)}")
    print()
    print("RETRIEVAL SCORES:")
    print(f"  Avg F1:        {avg_f1:.2%}")
    print(f"  Avg Precision: {avg_precision:.2%}")
    print(f"  Avg Recall:    {avg_recall:.2%}")
    print(f"  Perfect:       {sum(1 for r in successful_results if r['F1_Score'] == 1.0)}/{len(successful_results)}")
    print(f"  Zero:          {sum(1 for r in successful_results if r['F1_Score'] == 0.0)}/{len(successful_results)}")
    print()
    print(f"Avg Response Time: {sum(r['Response_Time_Seconds'] for r in results) / len(results):.2f}s")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run evaluation with retrieval scoring")
    parser.add_argument("--excel", default="data/document_analysis/Coding Challenge Evals.xlsx")
    parser.add_argument("--api-url", default="http://localhost:8000/query")
    parser.add_argument("--output", default="evaluation_results.xlsx")

    args = parser.parse_args()

    run_evaluation(
        excel_path=args.excel,
        api_url=args.api_url,
        output_path=args.output
    )
