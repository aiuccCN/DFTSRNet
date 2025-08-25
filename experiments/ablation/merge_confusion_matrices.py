#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to merge all confusion matrix TXT files
"""

import os
import glob
from datetime import datetime

def merge_confusion_matrices():
    """
    Merge all confusion matrix TXT files
    """
    # Define paths
    confusion_dir = r"e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\results\confusion_matrices"
    output_file = os.path.join(confusion_dir, "merged_confusion_matrices.txt")
    
    # Get all TXT files
    txt_files = glob.glob(os.path.join(confusion_dir, "*.txt"))
    txt_files = [f for f in txt_files if not f.endswith("merged_confusion_matrices.txt")]  # Exclude output file
    
    # Sort by filename
    txt_files.sort()
    
    print(f"Found {len(txt_files)} TXT files")
    
    # Merge files
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Write header
        outfile.write("="*80 + "\n")
        outfile.write("Ablation Experiment - All Confusion Matrices Merged Report\n")
        outfile.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        outfile.write(f"Number of files included: {len(txt_files)}\n")
        outfile.write("="*80 + "\n\n")
        
        # Read and write file contents one by one
        for i, txt_file in enumerate(txt_files, 1):
            filename = os.path.basename(txt_file)
            print(f"Processing file {i}/{len(txt_files)}: {filename}")
            
            outfile.write(f"\n{'='*60}\n")
            outfile.write(f"File {i}: {filename}\n")
            outfile.write(f"{'='*60}\n")
            
            try:
                with open(txt_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n")
            except Exception as e:
                outfile.write(f"Error reading file: {str(e)}\n")
                print(f"Error reading file {filename}: {str(e)}")
        
        # Write footer
        outfile.write("\n" + "="*80 + "\n")
        outfile.write("Merging completed\n")
        outfile.write("="*80 + "\n")
    
    print(f"\nMerging completed! Output file: {output_file}")
    print(f"Merged {len(txt_files)} files")
    
    # Display file size
    file_size = os.path.getsize(output_file)
    print(f"Output file size: {file_size:,} bytes")
    
    return output_file

def create_summary_table():
    """
    Create model performance summary table
    """
    confusion_dir = r"e:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\results\confusion_matrices"
    summary_file = os.path.join(confusion_dir, "performance_summary.txt")
    
    # Model list
    models = ['Base_CNN', '+AFBS', '+MGR', '+FiLM', 'Full', 'w_o_AFBS', 'w_o_FiLM', 'w_o_MGR']
    seeds = [20, 42, 97]
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Ablation Experiment Performance Summary Table\n")
        f.write("="*80 + "\n")
        f.write(f"Generation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Model Performance Overview:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Model Name':<15} {'Seed 20':<10} {'Seed 42':<10} {'Seed 97':<10} {'Average':<10}\n")
        f.write("-"*60 + "\n")
        
        for model in models:
            accuracies = []
            row = f"{model:<15}"
            
            for seed in seeds:
                filename = f"confusion_matrix_{model}_seed{seed}.txt"
                filepath = os.path.join(confusion_dir, filename)
                
                if os.path.exists(filepath):
                    # Can parse file to get accuracy, using placeholder for now
                    row += f"{'N/A':<10}"
                else:
                    row += f"{'Missing':<10}"
            
            row += f"{'N/A':<10}\n"
            f.write(row)
        
        f.write("-"*60 + "\n")
        f.write("\nNote: Specific accuracy rates need to be obtained from detailed result files\n")
    
    print(f"Performance summary table created: {summary_file}")
    return summary_file

if __name__ == "__main__":
    print("Starting to merge confusion matrix TXT files...")
    
    # Merge confusion matrix files
    merged_file = merge_confusion_matrices()
    
    # Create summary table
    summary_file = create_summary_table()

    print("\nProcessing completed!")
    print(f"Merged file: {merged_file}")
    print(f"Summary table: {summary_file}")