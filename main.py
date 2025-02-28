import argparse
import os
import subprocess
import zipfile
import yaml
import json
import sys
import git
import re
from typing import Dict, List, Optional, Any, Tuple

# Check for required packages and provide helpful error messages
try:
    import openai
except ImportError:
    print("Error: OpenAI package not installed. Install it with: pip install openai")
    print("Run: pip install openai")

try:
    import anthropic
except ImportError:
    print("Warning: Anthropic package not installed. Claude models won't be available.")
    print("To use Claude models, run: pip install anthropic")

try:
    import google.generativeai as genai
except ImportError:
    print("Warning: Google Generative AI package not installed. Gemini models won't be available.")
    print("To use Gemini models, run: pip install google-generativeai")

# Load configuration from config.yaml
def load_config():
    """Loads configuration settings from config.yaml."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: config.yaml not found at {config_path}")
        print("Please create a config.yaml file with your LLM settings.")
        print("Example config.yaml:")
        print("""
llm:
  default_provider: openai  # or anthropic, google
  openai:
    api_key: your_openai_api_key_here
    model: gpt-4-turbo-preview
  anthropic:
    api_key: your_anthropic_api_key_here
    model: claude-3-7-sonnet-20250219
  google:
    api_key: your_google_api_key_here
    model: gemini-pro
        """)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing config.yaml: {e}")
        sys.exit(1)

# Project management class
class Project:
    """Manages project directories and files."""
    def __init__(self, name):
        self.name = name
        self.path = os.path.join('projects', name)
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            
    def initialize_git(self):
        """Initialize git repository for the project if not already initialized."""
        if not os.path.exists(os.path.join(self.path, '.git')):
            repo = git.Repo.init(self.path)
            print(f"Initialized Git repository for project {self.name}")
            return repo
        return git.Repo(self.path)
            
    def commit_changes(self, message):
        """Commit all changes to the git repository."""
        try:
            repo = git.Repo(self.path)
            repo.git.add(A=True)  # Stage all changes, including new files
            try:
                repo.index.commit(message)  # Attempt to commit staged changes
                print(f"Changes committed: {message}")
            except git.exc.GitCommandError as e:
                if "nothing to commit" in str(e).lower():
                    print("No changes to commit")
                else:
                    raise  # Re-raise other Git errors for debugging
        except git.exc.InvalidGitRepositoryError:
            print("Git repository not initialized. Initializing now...")
            repo = self.initialize_git()
            repo.git.add(A=True)
            try:
                repo.index.commit(message)
                print(f"Initial commit: {message}")
            except git.exc.GitCommandError as e:
                if "nothing to commit" in str(e).lower():
                    print("No changes to commit")
                else:
                    raise

    def save_file(self, filename, content):
        """Saves content to a file in the project directory."""
        full_path = os.path.join(self.path, filename)
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved file: {filename}")

    def list_files(self, relative=True):
        """Returns a list of files in the project directory."""
        if not os.path.exists(self.path):
            return []
        
        all_files = []
        for root, _, files in os.walk(self.path):
            for file in files:
                if '.git' in root.split(os.sep):
                    continue  # Skip git files
                full_path = os.path.join(root, file)
                if relative:
                    all_files.append(os.path.relpath(full_path, self.path))
                else:
                    all_files.append(full_path)
        return all_files

    def get_file_content(self, filename):
        """Reads and returns the content of a file."""
        try:
            with open(os.path.join(self.path, filename), 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # If file is binary, return a placeholder
            return "[BINARY FILE]"
        except FileNotFoundError:
            print(f"Error: File {filename} not found")
            return None

    def get_project_summary(self, max_files=10, max_chars=1000):
        """Generate a summary of the project for context in LLM prompts."""
        files = self.list_files()
        summary = f"Project '{self.name}' contains {len(files)} files.\n\n"
        
        # Priority files that should always be included
        priority_files = [f for f in files if f.endswith(('package.json', 'README.md', '.env.example', 'app.js', 'index.js'))]
        
        # Other important files to consider
        js_files = [f for f in files if f.endswith('.js') and f not in priority_files]
        html_files = [f for f in files if f.endswith('.html')]
        css_files = [f for f in files if f.endswith('.css')]
        other_files = [f for f in files if f not in priority_files + js_files + html_files + css_files]
        
        # Sort files by importance and limit to max_files
        selected_files = priority_files + js_files + html_files + css_files + other_files
        selected_files = selected_files[:max_files]
        
        # Add file contents to summary
        for filename in selected_files:
            content = self.get_file_content(filename)
            if content:
                if len(content) > max_chars:
                    content = content[:max_chars] + "... [truncated]"
                summary += f"\n--- {filename} ---\n{content}\n"
        
        return summary

# LLM interaction class
class LLM:
    """Handles interactions with LLMs for code generation."""
    def __init__(self, config, provider=None):
        self.config = config
        
        # Use specified provider or default from config
        if provider:
            self.provider = provider.lower()
        elif 'default_provider' in config['llm']:
            self.provider = config['llm']['default_provider'].lower()
        else:
            # Fallback to the first available provider
            for provider in ['openai', 'anthropic', 'google']:
                if provider in config['llm']:
                    self.provider = provider
                    break
            else:
                raise ValueError("No valid LLM provider found in config")
        
        # Initialize the appropriate client
        if self.provider == 'openai':
            if 'openai' not in config['llm']:
                raise ValueError("OpenAI configuration missing from config.yaml")
                
            client_config = {}
            if 'api_key' in config['llm']['openai']:
                client_config['api_key'] = config['llm']['openai']['api_key']
            if 'base_url' in config['llm']['openai']:
                client_config['base_url'] = config['llm']['openai']['base_url']
            if 'organization' in config['llm']['openai']:
                client_config['organization'] = config['llm']['openai']['organization']
            
            self.client = openai.OpenAI(**client_config)
            self.model = config['llm']['openai'].get('model', 'gpt-4-turbo-preview')
            
        elif self.provider == 'anthropic':
            if 'anthropic' not in config['llm']:
                raise ValueError("Anthropic configuration missing from config.yaml")
                
            self.client = anthropic.Anthropic(
                api_key=config['llm']['anthropic']['api_key']
            )
            self.model = config['llm']['anthropic'].get('model', 'claude-3-7-sonnet-20250219')
            
        elif self.provider == 'google':
            if 'google' not in config['llm']:
                raise ValueError("Google configuration missing from config.yaml")
                
            genai.configure(api_key=config['llm']['google']['api_key'])
            self.client = genai
            self.model = config['llm']['google'].get('model', 'gemini-pro')
            
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
        print(f"Using {self.provider.capitalize()} model: {self.model}")

    def generate_code(self, prompt, system_message=None):
        """Generates code using the configured LLM based on the provided prompt."""
        try:
            if self.provider == 'openai':
                messages = []
                
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                    
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                )
                return response.choices[0].message.content
                
            elif self.provider == 'anthropic':
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=4000,
                    temperature=0.7,
                    system=system_message if system_message else "",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return message.content[0].text
                
            elif self.provider == 'google':
                generation_config = {
                    "temperature": 0.7,
                    "top_p": 1,
                    "top_k": 1,
                    "max_output_tokens": 8192,
                }
                
                model = self.client.GenerativeModel(model_name=self.model,
                                          generation_config=generation_config)
                
                prompt_parts = []
                if system_message:
                    prompt_parts.append(system_message)
                prompt_parts.append(prompt)
                
                response = model.generate_content(prompt_parts)
                return response.text
                
        except Exception as e:
            print(f"Error generating code with {self.provider}: {e}")
            return None

# Utility function to parse LLM responses for file operations
def parse_llm_response(response):
    """
    Parses the LLM response to extract file creation and modification instructions.
    """
    if not response:
        return []
        
    file_operations = []
    
    # Pattern for file creation: Create file: filename\n```language\ncode\n```
    create_matches = re.finditer(r'Create file: (.+?)\n```(?:\w+)?\n(.*?)\n```', response, re.DOTALL)
    for match in create_matches:
        filename, content = match.groups()
        filename = filename.strip()
        file_operations.append({
            'action': 'create',
            'filename': filename,
            'content': content
        })
    
    # Pattern for file modification: Modify file: filename\n```language\ncode\n```
    modify_matches = re.finditer(r'Modify file: (.+?)\n```(?:\w+)?\n(.*?)\n```', response, re.DOTALL)
    for match in modify_matches:
        filename, content = match.groups()
        filename = filename.strip()
        file_operations.append({
            'action': 'modify',
            'filename': filename,
            'content': content
        })
        
    # If no operations found but there's code blocks, try to extract them with filenames from surrounding text
    if not file_operations:
        # Look for filename followed by code block
        filename_code_matches = re.finditer(r'(?:^|\n)([a-zA-Z0-9_\-\.\/]+(?:\.[a-zA-Z0-9]+))(?::|)\s*\n```(?:\w+)?\n(.*?)\n```', response, re.DOTALL)
        for match in filename_code_matches:
            filename, content = match.groups()
            filename = filename.strip()
            file_operations.append({
                'action': 'create',
                'filename': filename,
                'content': content
            })
    
    return file_operations

# Command implementations
def create_project(args, config):
    """Creates a new project based on the user's description."""
    llm = LLM(config, args.provider)
    project = Project(args.project_name)
    
    # Check if project directory already exists and has files
    if project.list_files():
        if not args.force:
            print(f"Error: Project '{args.project_name}' already exists and contains files.")
            print("Use --force to overwrite the existing project.")
            return
        print(f"Warning: Overwriting existing project '{args.project_name}'")
    
    # Initialize git repository
    repo = project.initialize_git()
    
    # Construct system message
    system_message = (
        "You are an expert full-stack web developer tasked with generating code for a complete web application. "
        "Your code should be production-ready, well-structured, and follow best practices. "
        "For each file you create, format your response as:\n"
        "Create file: <filename>\n```<language>\n<code>\n```\n"
        "Ensure you provide ALL necessary files for the application to run, including package.json, "
        "README.md with setup instructions, and any configuration files."
    )
    
    # Construct prompt
    prompt = (
        f"Generate a full-stack web application that {args.description}.\n\n"
        "Use Node.js and Express for the backend.\n\n"
        "Include the following files:\n"
        "1. package.json with all dependencies\n"
        "2. A main app.js or index.js file\n"
        "3. Necessary HTML, CSS, and client-side JavaScript files\n"
        "4. A README.md with setup and usage instructions\n"
        "5. Any other required configuration files\n\n"
        "Make sure the application is complete and can be run with 'npm start'."
    )
    
    print("Generating project code... This may take a moment.")
    response = llm.generate_code(prompt, system_message)
    
    if not response:
        print("Failed to generate code. Check your API key and connection.")
        return
    
    file_operations = parse_llm_response(response)
    
    if file_operations:
        for operation in file_operations:
            if operation['action'] in ['create', 'modify']:
                project.save_file(operation['filename'], operation['content'])
        
        # Commit the initial code
        project.commit_changes("Initial project creation")
        
        print(f"\nProject '{args.project_name}' created successfully with {len(file_operations)} files.")
        print(f"To run the project: python main.py run --project-name {args.project_name}")
    else:
        print("Failed to parse LLM response. No file operations detected.")
        print("Raw response:")
        print("--------------------")
        print(response)
        print("--------------------")

def add_feature(args, config):
    """Adds a new feature to an existing project."""
    llm = LLM(config, args.provider)
    project = Project(args.project_name)
    
    # Check if project exists
    if not os.path.exists(project.path) or not project.list_files():
        print(f"Error: Project '{args.project_name}' doesn't exist or is empty.")
        return
    
    # Get project summary for context
    project_summary = project.get_project_summary()
    
    # Construct system message
    system_message = (
        "You are an expert full-stack web developer tasked with adding features to an existing web application. "
        "Analyze the provided project files and implement the requested feature in a way that integrates "
        "seamlessly with the existing codebase. Format your changes as:\n"
        "Create file: <filename>\n```<language>\n<code>\n```\n"
        "Or for modifications:\n"
        "Modify file: <filename>\n```<language>\n<new content>\n```\n"
        "Only include the files you need to create or modify."
    )
    
    # Construct prompt
    prompt = (
        f"Add the following feature to the existing project: {args.feature}\n\n"
        "Here is the current state of the project:\n\n"
        f"{project_summary}\n\n"
        "Provide the necessary code changes to implement this feature. "
        "Make sure your changes integrate well with the existing code."
    )
    
    print("Generating feature code... This may take a moment.")
    response = llm.generate_code(prompt, system_message)
    
    if not response:
        print("Failed to generate code. Check your API key and connection.")
        return
    
    file_operations = parse_llm_response(response)
    
    if file_operations:
        for operation in file_operations:
            if operation['action'] in ['create', 'modify']:
                project.save_file(operation['filename'], operation['content'])
        
        # Commit the changes
        project.commit_changes(f"Added feature: {args.feature}")
        
        print(f"\nFeature '{args.feature}' added successfully with {len(file_operations)} file changes.")
    else:
        print("Failed to parse LLM response. No file operations detected.")
        print("Raw response:")
        print("--------------------")
        print(response)
        print("--------------------")

def fix_bug(args, config):
    """Fixes a bug in an existing project."""
    llm = LLM(config, args.provider)
    project = Project(args.project_name)
    
    # Check if project exists
    if not os.path.exists(project.path) or not project.list_files():
        print(f"Error: Project '{args.project_name}' doesn't exist or is empty.")
        return
    
    # Get project summary for context
    project_summary = project.get_project_summary()
    
    # Construct system message
    system_message = (
        "You are an expert full-stack web developer tasked with fixing bugs in an existing web application. "
        "Analyze the provided project files, identify the cause of the bug described, and implement a solution. "
        "Format your changes as:\n"
        "Modify file: <filename>\n```<language>\n<new content>\n```\n"
        "Only include the files you need to modify."
    )
    
    # Construct prompt
    prompt = (
        f"Fix the following bug in the existing project: {args.bug_description}\n\n"
        "Here is the current state of the project:\n\n"
        f"{project_summary}\n\n"
        "Provide the necessary code changes to fix this bug. "
        "Make sure your solution addresses the root cause."
    )
    
    print("Generating bug fix... This may take a moment.")
    response = llm.generate_code(prompt, system_message)
    
    if not response:
        print("Failed to generate code. Check your API key and connection.")
        return
    
    file_operations = parse_llm_response(response)
    
    if file_operations:
        for operation in file_operations:
            if operation['action'] in ['create', 'modify']:
                project.save_file(operation['filename'], operation['content'])
        
        # Commit the changes
        project.commit_changes(f"Fixed bug: {args.bug_description}")
        
        print(f"\nBug '{args.bug_description}' fixed successfully with {len(file_operations)} file changes.")
    else:
        print("Failed to parse LLM response. No file operations detected.")
        print("Raw response:")
        print("--------------------")
        print(response)
        print("--------------------")

def run_project(args, config):
    """Runs the project locally using Node.js."""
    project = Project(args.project_name)
    
    # Check if project exists
    if not os.path.exists(project.path) or not project.list_files():
        print(f"Error: Project '{args.project_name}' doesn't exist or is empty.")
        return
    
    # Check if package.json exists
    if not os.path.exists(os.path.join(project.path, 'package.json')):
        print("Error: package.json not found. Can't run the project.")
        return
    
    print(f"Running project '{args.project_name}'...")
    
    try:
        print("Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=project.path, check=True)
        
        print("\nStarting the application...")
        subprocess.run(["npm", "start"], cwd=project.path)
    except subprocess.CalledProcessError as e:
        print(f"Error running project: {e}")
    except FileNotFoundError:
        print("Error: npm command not found. Make sure Node.js is installed.")

def deploy_project(args, config):
    """Packages the project for deployment."""
    project = Project(args.project_name)
    
    # Check if project exists
    if not os.path.exists(project.path) or not project.list_files():
        print(f"Error: Project '{args.project_name}' doesn't exist or is empty.")
        return
    
    zip_filename = f"{project.name}.zip"
    
    try:
        with zipfile.ZipFile(zip_filename, "w") as zipf:
            for root, dirs, files in os.walk(project.path):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Skip .git directory
                    if '.git' in file_path.split(os.sep):
                        continue
                    zipf.write(file_path, os.path.relpath(file_path, os.path.dirname(project.path)))
        
        print(f"Project packaged into '{zip_filename}'.")
        print(f"Deployment package size: {os.path.getsize(zip_filename) / (1024*1024):.2f} MB")
        print("\nDeployment instructions:")
        print("1. Upload the ZIP file to your hosting provider")
        print("2. Extract the contents to your server")
        print("3. Run 'npm install' to install dependencies")
        print("4. Run 'npm start' or set up a process manager like PM2")
    except Exception as e:
        print(f"Error packaging project: {e}")

def list_providers(args, config):
    """Lists all available LLM providers in the config file."""
    print("Available LLM providers:")
    
    if 'llm' not in config:
        print("No LLM providers configured in config.yaml")
        return
    
    if 'default_provider' in config['llm']:
        default = config['llm']['default_provider']
        print(f"Default provider: {default}")
    
    providers = []
    
    if 'openai' in config['llm']:
        model = config['llm']['openai'].get('model', 'gpt-4-turbo-preview')
        providers.append(f"openai: {model}")
    
    if 'anthropic' in config['llm']:
        model = config['llm']['anthropic'].get('model', 'claude-3-7-sonnet-20250219')
        providers.append(f"anthropic: {model}")
    
    if 'google' in config['llm']:
        model = config['llm']['google'].get('model', 'gemini-pro')
        providers.append(f"google: {model}")
    
    if providers:
        for provider in providers:
            print(f"- {provider}")
    else:
        print("No providers configured. Please update your config.yaml file.")

def setup_config(args):
    """Creates or updates the config.yaml file."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    
    # Check if config file already exists
    if os.path.exists(config_path) and not args.force:
        print(f"Config file already exists at {config_path}")
        print("Use --force to overwrite the existing config.")
        return
    
    # Create default config
    config = {
        'llm': {
            'default_provider': 'openai',
            'openai': {
                'api_key': 'your_openai_api_key_here',
                'model': 'gpt-4-turbo-preview'
            },
            'anthropic': {
                'api_key': 'your_anthropic_api_key_here',
                'model': 'claude-3-7-sonnet-20250219'
            },
            'google': {
                'api_key': 'your_google_api_key_here',
                'model': 'gemini-pro'
            }
        }
    }
    
    # Write config to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created config file at {config_path}")
    print("Please edit the file to add your API keys.")

# Main CLI setup
def main():
    """Sets up and runs the CLI with available commands."""
    parser = argparse.ArgumentParser(description="FullStackGPT - AI-assisted web development tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Version
    parser.add_argument('--version', action='version', version='FullStackGPT v0.2.0')

    # Provider argument that can be used with any command
    provider_help = "LLM provider to use (openai, anthropic, google)"

    # Create command
    create_parser = subparsers.add_parser('create', help="Create a new web application")
    create_parser.add_argument('description', type=str, help="Description of the application")
    create_parser.add_argument('--project-name', type=str, required=True, help="Name of the project")
    create_parser.add_argument('--force', action='store_true', help="Force overwrite if project exists")
    create_parser.add_argument('--provider', type=str, help=provider_help)

    # Add feature command
    add_parser = subparsers.add_parser('add', help="Add a feature to an existing project")
    add_parser.add_argument('feature', type=str, help="Description of the feature to add")
    add_parser.add_argument('--project-name', type=str, required=True, help="Name of the project")
    add_parser.add_argument('--provider', type=str, help=provider_help)

    # Fix bug command
    fix_parser = subparsers.add_parser('fix', help="Fix a bug in an existing project")
    fix_parser.add_argument('bug_description', type=str, help="Description of the bug to fix")
    fix_parser.add_argument('--project-name', type=str, required=True, help="Name of the project")
    fix_parser.add_argument('--provider', type=str, help=provider_help)

    # Run command
    run_parser = subparsers.add_parser('run', help="Run the project locally")
    run_parser.add_argument('--project-name', type=str, required=True, help="Name of the project")

    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help="Package the project for deployment")
    deploy_parser.add_argument('--project-name', type=str, required=True, help="Name of the project")
    
    # List providers command
    list_parser = subparsers.add_parser('list-providers', help="List available LLM providers")
    
    # Setup config command
    setup_parser = subparsers.add_parser('setup', help="Create or update the config.yaml file")
    setup_parser.add_argument('--force', action='store_true', help="Force overwrite if config exists")

    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Handle setup command (doesn't require config)
    if args.command == 'setup':
        setup_config(args)
        return
    
    # Load configuration for other commands
    try:
        config = load_config()
    except Exception as e:
        print(f"Error loading configuration: {e}")
        print("You may need to run 'python main.py setup' to create a config file.")
        return
    
    # Execute the appropriate command
    if args.command == 'create':
        create_project(args, config)
    elif args.command == 'add':
        add_feature(args, config)
    elif args.command == 'fix':
        fix_bug(args, config)
    elif args.command == 'run':
        run_project(args, config)
    elif args.command == 'deploy':
        deploy_project(args, config)
    elif args.command == 'list-providers':
        list_providers(args, config)

if __name__ == '__main__':
    main()