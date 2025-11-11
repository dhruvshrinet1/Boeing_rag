"""
Evaluation Script for Boeing 737 Manual RAG Service
Runs all questions from the evaluation Excel file and generates results.
"""
import pandas as pd
import requests
import time
from datetime import datetime
import json


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
    print("Boeing 737 Manual RAG Service - Evaluation")
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
        expected_pages = row['Page reference']

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

                print(f"✓ Success ({elapsed_time:.2f}s)")
                print(f"  Pages returned: {actual_pages}")
                print(f"  Expected pages: {expected_pages}")

                # Store results
                results.append({
                    'Question_Number': question_num,
                    'Question': question,
                    'Expected_Answer': expected_answer,
                    'Expected_Pages': str(expected_pages),
                    'API_Answer': actual_answer,
                    'API_Pages': str(actual_pages),
                    'API_Pages_List': actual_pages,
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
                    'Expected_Pages': str(expected_pages),
                    'API_Answer': f"ERROR: {response.text[:100]}",
                    'API_Pages': 'N/A',
                    'API_Pages_List': [],
                    'Response_Time_Seconds': round(elapsed_time, 2),
                    'Status': f'Error_{response.status_code}'
                })

        except Exception as e:
            print(f"✗ Exception: {str(e)}")

            results.append({
                'Question_Number': question_num,
                'Question': question,
                'Expected_Answer': expected_answer,
                'Expected_Pages': str(expected_pages),
                'API_Answer': f"ERROR: {str(e)}",
                'API_Pages': 'N/A',
                'API_Pages_List': [],
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
                'Average Response Time (s)',
                'Evaluation Date',
                'API URL'
            ],
            'Value': [
                len(results),
                sum(1 for r in results if r['Status'] == 'Success'),
                sum(1 for r in results if r['Status'] != 'Success'),
                round(sum(r['Response_Time_Seconds'] for r in results) / len(results), 2) if results else 0,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                api_url
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"✓ Results saved to {output_path}")
    print()

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    successful = sum(1 for r in results if r['Status'] == 'Success')
    print(f"Total Questions:        {len(results)}")
    print(f"Successful Responses:   {successful}")
    print(f"Failed Responses:       {len(results) - successful}")
    print(f"Average Response Time:  {sum(r['Response_Time_Seconds'] for r in results) / len(results):.2f}s")
    print("=" * 80)
    print()

    # Show sample results
    if successful > 0:
        print("Sample Results (First Question):")
        print("-" * 80)
        first_success = next(r for r in results if r['Status'] == 'Success')
        print(f"Question: {first_success['Question'][:100]}...")
        print(f"\nExpected Pages: {first_success['Expected_Pages']}")
        print(f"API Pages:      {first_success['API_Pages']}")
        print(f"\nAnswer Preview: {first_success['API_Answer'][:200]}...")
        print("-" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Boeing RAG evaluation")
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
