import os
import shutil
import tempfile
import stat
from git import Repo

SUPPORTED_EXTENSIONS = {'.py', '.js', '.ts', '.java', '.cpp', '.h', '.cs', '.go'}

class CodeParser:
    def __init__(self):
        pass

    def clone_repository(self, repo_url: str) -> str:
        temp_dir = tempfile.mkdtemp(prefix="ai_repo_")
        print(f"\n[GIT] Cloning repository: {repo_url}")
        Repo.clone_from(repo_url, temp_dir, depth=1)
        return temp_dir

    def chunk_file(self, file_path: str, relative_path: str) -> list[dict]:
        chunks = []
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f" Read Error on {relative_path}: {str(e)}")
            return chunks

        total_lines = len(lines)
        if total_lines == 0:
            return chunks

        print(f" Chunking: {relative_path} ({total_lines} lines)")
        chunk_size = 25
        overlap = 5
        start_idx = 0
        
        while start_idx < total_lines:
            end_idx = min(start_idx + chunk_size, total_lines)
            chunk_lines = lines[start_idx:end_idx]
            code_content = "".join(chunk_lines)
            
            chunks.append({
                "content": code_content,
                "metadata": {
                    "file_path": relative_path,
                    "start_line": start_idx + 1,
                    "end_line": end_idx,
                    "context": f"Lines {start_idx+1}-{end_idx}",
                    "language": "java" if relative_path.endswith(".java") else "code"
                }
            })
            start_idx += (chunk_size - overlap)
            
        return chunks

    def process_repository(self, repo_path: str) -> list[dict]:
        all_chunks = []
        print("\n--- STARTING CODEBASE SCAN ---")
        
        for root, dirs, files in os.walk(repo_path):
            if '.git' in root.split(os.sep):
                continue
                
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, repo_path)
                    
                    file_chunks = self.chunk_file(full_path, relative_path)
                    all_chunks.extend(file_chunks)
        
        print(f"--- SCAN COMPLETE: Generated {len(all_chunks)} total chunks ---")
        
        def remove_readonly(func, path, excinfo):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        try:
            shutil.rmtree(repo_path, onerror=remove_readonly)
            print("[GIT] Temporary folder deleted successfully.")
        except Exception as e:
            print(f"[⚠️ Warning] Could not clear temp folder: {e}")
            
        return all_chunks