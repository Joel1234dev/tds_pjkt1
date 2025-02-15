from flask import Flask, request
import subprocess
from datetime import datetime
import calendar
import os
import json
import re
import base64
import sqlite3
import time
import requests

app = Flask(__name__)

def format_action(file_path='/data/format.md'):
    try:
        if not os.path.exists('package.json'):
            init_cmd = ['npm', 'init', '-y']
            subprocess.run(init_cmd, capture_output=True, text=True)
        
        install_cmd = ['npm', 'install', 'prettier@3.4.2']
        subprocess.run(install_cmd, capture_output=True, text=True)
        
        prettier_path = './node_modules/.bin/prettier'
        cmd_parts = [prettier_path, '--write', file_path]
        result = subprocess.run(cmd_parts, capture_output=True, text=True)
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except Exception as e:
        return {'error': str(e)}
    
def count_the_number_of_weekdays_action(weekday, from_file='/data/dates.txt', to_file='/data/dates-weekday.txt'):
    try:
        os.makedirs(os.path.dirname(from_file), exist_ok=True)
        
        if not os.path.exists(from_file):
            return {'error': f'Input file {from_file} does not exist'}
            
        weekday_count = 0
        
        with open(from_file, 'r') as f:
            for line in f:
                date_str = line.strip()
                try:
                    for fmt in [
                        '%b %d, %Y',           # Mar 13, 2009
                        '%Y-%m-%d',            # 2012-11-17
                        '%d-%b-%Y',            # 28-Jul-2008
                        '%Y/%m/%d %H:%M:%S',   # 2022/04/28 07:27:13
                        '%d/%m/%Y',            # 13/03/2009
                        '%Y.%m.%d',            # 2012.11.17
                        '%d.%m.%Y',            # 28.07.2008
                        '%m/%d/%Y',            # 03/13/2009
                        '%Y-%m-%d %H:%M',      # 2022-04-28 07:27
                        '%d-%m-%Y'             # 28-07-2008
                    ]:
                        try:
                            date = datetime.strptime(date_str, fmt)
                            if date.weekday() == list(calendar.day_name).index(weekday.capitalize()):
                                weekday_count += 1
                            break
                        except ValueError:
                            continue
                except Exception as e:
                    print(f"Error parsing date {date_str}: {str(e)}")
                    continue
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        
        # Write count to file
        with open(to_file, 'w') as f:
            f.write(str(weekday_count))
            
        return {'message': f'Successfully counted {weekday_count} {weekday}s'}
        
    except Exception as e:
        return {'error': str(e)}

def sort_contacts_action(from_file='/data/contacts.json', to_file='/data/contacts-sorted.json'):
    with open(from_file, 'r') as f:
        contacts = json.load(f)
    
    sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))
    
    os.makedirs('/data', exist_ok=True)
    with open(to_file, 'w') as f:
        json.dump(sorted_contacts, f, indent=2)

def cp_most_recent_logs_action(to_file='/data/logs-recent.txt', log_dir='/data/logs', top_n=10):
    try:
        log_files = []
        for file in os.listdir(log_dir):
            if file.endswith('.log'):
                path = os.path.join(log_dir, file)
                log_files.append((path, os.path.getmtime(path)))
        
        log_files.sort(key=lambda x: x[1], reverse=True)
        
        print(log_files)

        recent_logs = log_files[:top_n]
        
        os.makedirs('/data', exist_ok=True)
        
        with open(to_file, 'w') as outfile:
            for log_path, _ in recent_logs:
                with open(log_path, 'r') as logfile:
                    first_line = logfile.readline().strip()
                    outfile.write(first_line + '\n')
    except Exception as e:
        print(f"Error processing log files: {str(e)}")
        raise

def index_docs_action(docs_dir='/data/docs', index_file='/data/docs/index.json', elem_symbol='#'):
    try:
        index = {}
        
        for root, _, files in os.walk(docs_dir):
            for file in files:
                if file.endswith('.md'):
                    full_path = os.path.join(root, file)
                    
                    rel_path = os.path.relpath(full_path, docs_dir)
                    
                    with open(full_path, 'r', encoding='utf-8') as f:
                        title = None
                        for line in f:
                            line = line.strip()
                            if line.startswith(elem_symbol + ' '):
                                title = line[len(elem_symbol) + 1:].strip()
                                break
                        
                        if title:
                            index[rel_path] = title
        
        os.makedirs('/data', exist_ok=True)
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
            
    except Exception as e:
        print(f"Error creating docs index: {str(e)}")
        raise

def extract_sender_action(from_file='/data/email.txt', to_file='/data/email-sender.txt'):
    try:
        with open(from_file, 'r') as f:
            email_content = f.read()

        llm_prompt = "Extract only the sender's email address from this text blob:\n\n" + email_content
        sender_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', llm_prompt)
        sender_email = sender_match.group(1) if sender_match else ""
        
        os.makedirs('/data', exist_ok=True)
        with open(to_file, 'w') as f:
            f.write(sender_email)
            
    except Exception as e:
        print(f"Error extracting sender email: {str(e)}")
        raise

def extract_card_number_action(from_file='/data/credit-card.png', to_file='/data/credit-card.txt'):
    try:
        with open(from_file, 'rb') as f:
            image_data = f.read()
            
        llm_prompt = "Extract only the credit card number from this image, with no spaces or other characters"

        llm_response = get_llm_response(llm_prompt, image_data)

        card_match = re.search(r'\d{16}', llm_response)
        card_number = card_match.group(0) if card_match else ""
        
        os.makedirs('/data', exist_ok=True)
        with open(to_file, 'w') as f:
            f.write(card_number)
            
    except Exception as e:
        print(f"Error extracting credit card number: {str(e)}")
        raise

TOKEN = os.environ["AIPROXY_TOKEN"]

def find_similar_comments_action(from_file='/data/comments.txt', to_file='/data/comments-similar.txt'):
    try:
        with open(from_file, 'r', encoding='utf-8') as f:
            comments = [line.strip() for line in f.readlines()]
            
        if len(comments) < 2:
            raise ValueError("Need at least 2 comments to find similar pair")
        
        embeddings = [get_embedding(comment) for comment in comments]
            
        max_similarity = -1
        most_similar = (0, 1)
        
        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                dot_product = sum(a * b for a, b in zip(embeddings[i], embeddings[j]))
                norm_i = sum(x * x for x in embeddings[i]) ** 0.5
                norm_j = sum(x * x for x in embeddings[j]) ** 0.5
                similarity = dot_product / (norm_i * norm_j)
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar = (i, j)
        
        os.makedirs('/data', exist_ok=True)
        with open(to_file, 'w', encoding='utf-8') as f:
            f.write(comments[most_similar[0]] + '\n')
            f.write(comments[most_similar[1]])
            
    except Exception as e:
        print(f"Error finding similar comments: {str(e)}")
        raise

def execute_sql_action(db_path='/data/ticket-sales.db', to_file='/data/ticket-sales-gold.txt', sql_query='SELECT SUM(units * price) FROM tickets WHERE type = "Gold"'):
    try:        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(sql_query)
        
        total_sales = cursor.fetchone()[0]
        if total_sales is None:
            total_sales = 0
            
        conn.close()
        
        os.makedirs('/data', exist_ok=True)
        with open(to_file, 'w') as f:
            f.write(str(total_sales))
            
    except Exception as e:
        print(f"Error calculating gold ticket sales: {str(e)}")
        raise

def commands_action(objective=None, initial_command=None):
    try:
        command_history = []
        max_iterations = 50
        iteration = 0
        result = None

        if initial_command:
            os.makedirs('/data', exist_ok=True)
            result = subprocess.run(initial_command, shell=True, capture_output=True, text=True, cwd='/data')
            
            command_history.append({
                'command': initial_command,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'success': result.returncode == 0
            })

            if not objective:
                return {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }

        while iteration < max_iterations:
            system_prompt = """You are a command line assistant focused on executing commands in the /data directory.

IMPORTANT RULES:
1. ALL commands must be executed in the /data directory
3. ALWAYS create directories before trying to write files
4. Keep commands simple and atomic - one operation per command
5. If the objective is met, respond with: <TASK_COMPLETE>DONE</TASK_COMPLETE>
6. Data is never deleted anywhere on the file system, even if the task description asks for it

Command History:"""

            for entry in command_history:
                status = "✓ Success" if entry['success'] else "✗ Failed"
                system_prompt += f"\n\n[{status}] Command: {entry['command']}"
                if entry['stdout']:
                    system_prompt += f"\nOutput:\n{entry['stdout'].strip()}"
                if entry['stderr']:
                    system_prompt += f"\nErrors:\n{entry['stderr'].strip()}"
                system_prompt += f"\nReturn code: {entry['returncode']}"

            if command_history and not command_history[-1]['success']:
                system_prompt += "\n\nWARNING: Previous command failed. Please try a different approach."
            
            if objective:
                system_prompt += f"\n\nCURRENT OBJECTIVE: {objective}\n"
                system_prompt += "\nWhat single command should I run next to help achieve this objective? If the objective is complete or cannot be achieved, respond with <TASK_COMPLETE>DONE</TASK_COMPLETE>. Otherwise, enclose the command in ```bash``` tags."
            else:
                system_prompt += "\n\nWhat command would you like me to run? Make sure to enclose the command in ```bash``` tags."

            while True:
                try:
                    llm_response = get_llm_response([
                        {"role": "system", "content": system_prompt}
                    ])
                    break
                except Exception as e:
                    error_str = str(e)
                    if "rate_limit_exceeded" in error_str:
                        match = re.search(r'try again in (\d+\.?\d*)s', error_str)
                        if match:
                            timeout = float(match.group(1))
                            time.sleep(timeout)
                            continue
                    return {
                        'stdout': '',
                        'stderr': f'Error getting LLM response: {error_str}',
                        'returncode': 1
                    }

            command = llm_response.strip()
            
            if '<TASK_COMPLETE>' in command and '</TASK_COMPLETE>' in command:
                break

            if '```' in command:
                command = re.search(r'```(?:bash)?\s*(.*?)\s*```', command, re.DOTALL)
                if command:
                    command = command.group(1).strip()
                else:
                    command = llm_response.strip()


            print(command)
            if objective and command.upper() == "DONE":
                break

            if command_history and command == command_history[-1]['command']:
                break

            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            command_history.append({
                'command': command,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode,
                'success': result.returncode == 0
            })

            if not objective:
                return {
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'returncode': result.returncode
                }
                
            if len(command_history) >= 3:
                last_three = command_history[-3:]
                if all(not entry['success'] for entry in last_three):
                    break
                
            iteration += 1

        if objective:
            summary_prompt = f"""Based on the command history:
{system_prompt}
Please analyze the command history and provide a clear summary of the results related to the objective. For example, if the objective was to count files, include the total number found. Focus on key metrics and outcomes that directly address the original objective.
"""

            while True:
                try:
                    final_summary = get_llm_response([
                        {"role": "system", "content": summary_prompt}
                    ])
                    break
                except Exception as e:
                    error_str = str(e)
                    if "rate_limit_exceeded" in error_str:
                        match = re.search(r'try again in (\d+\.?\d*)s', error_str)
                        if match:
                            timeout = float(match.group(1))
                            time.sleep(timeout)
                            continue
                    return {
                        'stdout': '',
                        'stderr': f'Error getting summary: {error_str}',
                        'returncode': 1
                    }

            had_success = any(entry['success'] for entry in command_history)
            return {
                'stdout': final_summary,
                'stderr': '' if had_success else 'All commands failed',
                'returncode': 0 if had_success and 'error' not in final_summary.lower() else 1
            }

    except Exception as e:
        return {
            'stdout': '',
            'stderr': str(e),
            'returncode': 1
        }

def get_llm_response(text, image=None):
    messages = []
    if isinstance(text, list):
        messages = text
    else:
        content = []
        content.append({
            "type": "text",
            "text": text
        })
        
        if image is not None:
            base64_image = base64.b64encode(image).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })
            
        messages = [{
            "role": "user", 
            "content": content
        }]

    try:
        response = requests.post(
            "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {TOKEN}"
            },
            json={
                "model": "gpt-4o-mini",
                "messages": messages if isinstance(text, list) else messages,
                "max_tokens": 200,
                "temperature": 0.7
            }
        )
        
        response_json = response.json()

        print(response_json)
        if 'choices' not in response_json:
            raise Exception(f"Invalid API response: {response_json}")
            
        return response_json["choices"][0]["message"]["content"].strip()
        
    except Exception as e:
        raise Exception(f"Error calling LLM API: {str(e)}")

def get_embedding(text):    
    response = requests.post(
        "http://aiproxy.sanand.workers.dev/openai/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {TOKEN}"
        },
        json={
            "model": "text-embedding-3-small",
            "input": text
        }
    )
    
    return response.json()["data"][0]["embedding"]

def run_command_action(command=None):
    try:
        print(command)
        if not command:
            return {
                'stdout': '',
                'stderr': 'No command provided',
                'returncode': 1
            }
            
        os.makedirs('/data', exist_ok=True)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        return {
            'stdout': result.stdout,
            'stderr': result.stderr, 
            'returncode': result.returncode
        }
        
    except Exception as e:
        print(f"Error running command: {str(e)}")
        raise

@app.route('/run', methods=['POST'])
def run_task():
    task = request.args.get('task')
    if not task:
        return 'Missing task parameter', 400
    
    try:
        system_prompt = f"""
<system>
You are a helpful assistant that can execute actions on a local machine.

Your response must be in the following format:
<action name="action_name">
<parameter name="param1">value1</parameter>
<parameter name="param2">value2</parameter>
</action>

Available actions:

1. format_action
- Formats a file using Prettier
- Parameters:
  - file_path (string, default: "/data/format.md"): Path to file to format

2. count_the_number_of_weekdays_action  
- Counts occurrences of a weekday in dates file
- Parameters:
  - weekday (string): Name of weekday to count
  - from_file (string, default: "/data/dates.txt"): Input file path
  - to_file (string, default: "/data/dates-weekday.txt"): Output file path

3. sort_contacts_action
- Sorts contacts by last name then first name
- Parameters:
  - from_file (string, default: "/data/contacts.json"): Input JSON file
  - to_file (string, default: "/data/contacts-sorted.json"): Output file

4. cp_most_recent_logs_action
- Copies first line from N most recent logs
- Parameters:
  - to_file (string, default: "/data/logs-recent.txt"): Output file
  - log_dir (string, default: "/data/logs"): Logs directory
  - top_n (integer, default: 10): Number of files to process

5. index_docs_action
- Indexes markdown docs by title
- Parameters:
  - docs_dir (string, default: "/data/docs"): Docs directory
  - index_file (string, default: "/data/docs/index.json"): Output index
  - elem_symbol (string, default: "#"): Title symbol

6. extract_sender_action
- Extracts sender email from email file
- Parameters:
  - from_file (string, default: "/data/email.txt"): Input email
  - to_file (string, default: "/data/email-sender.txt"): Output file

7. extract_card_number_action
- Extracts credit card number from image
- Parameters:
  - from_file (string, default: "/data/credit-card.png"): Input image
  - to_file (string, default: "/data/credit-card.txt"): Output file

8. find_similar_comments_action
- Finds most similar comment pair
- Parameters:
  - from_file (string, default: "/data/comments.txt"): Input comments
  - to_file (string, default: "/data/comments-similar.txt"): Output file

9. execute_sql_action
- Executes a SQL query on a database
- Parameters:
  - db_path (string, default: "/data/ticket-sales.db"): Database path
  - to_file (string, default: "/data/ticket-sales-gold.txt"): Output file
  - sql_query (string, default: "SELECT SUM(units * price) FROM tickets WHERE type = "Gold""): SQL query to execute

10. commands_action
- Executes shell commands to achieve an objective.
- Its Agentic in nature, so it does a agentic loop until the objective is met or it runs out of iterations.
- Its prone to hallucinate, so you need to be careful with the commands you give it.
- Only use for big workflows, not for single commands.
- Can potentially perform operations like:
  - Editing files (e.g. using vim, nano, or other text editors)
  - Fetching data from APIs (e.g. using curl or wget)
  - Git operations (e.g. cloning repos and making commits)
  - Running SQL queries on databases (e.g. SQLite/DuckDB)
  - Web scraping (e.g. using curl, wget, or scrapy)
  - Image processing (e.g. using ImageMagick)
  - Audio transcription (e.g. using speech-to-text tools)
  - Converting between formats (e.g. Markdown to HTML with pandoc)
  - Processing data files (e.g. CSV/JSON with jq, awk, or sed)
- Parameters:
  - objective (string): Goal to achieve with commands
  - initial_command (string, optional): Initial command to run

11. run_command_action
- Executes a shell command
- Only use for single commands, not for big workflows.
- Can be used to just run a command and not return anything, better than commands_action because it runs one command only.
- Parameters:
  - command (string): Command to execute
</system>

<user>
  <task>
    {task}
  </task>
</user>
"""
        try:
            llm_response = get_llm_response(system_prompt)
        except Exception as e:
            return f"Error getting LLM response: {str(e)}", 500
        
        print(llm_response)
        action_match = re.search(r'<action name="([^"]+)">', llm_response)
        if not action_match:
            raise ValueError("No valid action found in response")
            
        action_name = action_match.group(1)
        
        params = {}
        param_matches = re.finditer(r'<parameter name="([^"]+)">([^<]+)</parameter>', llm_response)
        for match in param_matches:
            params[match.group(1)] = match.group(2)
            
        if action_name == "format_action":
            result = format_action(**params)
        elif action_name == "count_the_number_of_weekdays_action":
            result = count_the_number_of_weekdays_action(**params)
        elif action_name == "sort_contacts_action":
            sort_contacts_action(**params)
            result = "Success"
        elif action_name == "cp_most_recent_logs_action":
            cp_most_recent_logs_action(**params)
            result = "Success"
        elif action_name == "index_docs_action":
            index_docs_action(**params)
            result = "Success"
        elif action_name == "extract_sender_action":
            extract_sender_action(**params)
            result = "Success"
        elif action_name == "extract_card_number_action":
            extract_card_number_action(**params)
            result = "Success"
        elif action_name == "find_similar_comments_action":
            find_similar_comments_action(**params)
            result = "Success"
        elif action_name == "execute_sql_action":
            execute_sql_action(**params)
            result = "Success"
        elif action_name == "commands_action":
            result = commands_action(**params)
        elif action_name == "run_command_action":
            result = run_command_action(**params)
        else:
            raise ValueError(f"Unknown action: {action_name}")
        

        return result, 200
    except ValueError as e:
        return str(e), 400
    except Exception as e:
        return str(e), 500

@app.route('/read', methods=['GET'])
def read_file():
    file_path = request.args.get('path')
    if not file_path:
        return 'Missing path parameter', 400
        
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            return content, 200
    except FileNotFoundError:
        return '', 404
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True) 