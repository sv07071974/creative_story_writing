import gradio as gr
import requests
import json
from typing import List, Dict

# Define the API endpoint for Ollama local server
OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}

class StoryAssistant:
    def __init__(self):
        self.models = [
            "mistral:latest",
            "llama3.2:latest",
            "gemma:7b",
            "qwen2.5-coder:3b"
        ]
        
        self.genres = [
            "Fantasy",
            "Science Fiction",
            "Mystery",
            "Romance",
            "Horror",
            "Literary Fiction",
            "Historical Fiction",
            "Adventure",
            "Thriller",
            "Comedy"
        ]
        
        self.writing_styles = [
            "Descriptive",
            "Minimalist",
            "Stream of Consciousness",
            "Lyrical",
            "Suspenseful",
            "Humorous",
            "Dark and Moody",
            "Fast-paced"
        ]
        
        self.assistance_types = [
            "Generate Story Idea",
            "Develop Plot Outline",
            "Create Character Profile",
            "Write Opening Paragraph",
            "Generate Dialogue",
            "Describe Setting",
            "Write Complete Short Story"
        ]

    def generate_content(self,
                        model_name: str,
                        assistance_type: str,
                        genre: str,
                        writing_style: str,
                        additional_prompt: str,
                        word_limit: int,
                        tone: str) -> str:
        """
        Generate creative writing content based on user parameters
        """
        try:
            # Create base system prompt for creative writing
            system_prompt = f"""You are a creative writing assistant specializing in {genre} with a 
            {writing_style} style. Create {tone} content that is engaging and original."""
            
            # Customize user prompt based on assistance type
            prompts = {
                "Generate Story Idea": f"Generate a unique story idea for a {genre} story with approximately {word_limit} words.",
                
                "Develop Plot Outline": f"""Create a detailed plot outline for a {genre} story. 
                Include key plot points, conflict, and resolution.""",
                
                "Create Character Profile": f"""Develop a detailed character profile including personality, 
                background, motivations, and conflicts for a {genre} story.""",
                
                "Write Opening Paragraph": f"""Write an engaging opening paragraph for a {genre} story 
                using a {writing_style} style. Set the tone and hook the reader.""",
                
                "Generate Dialogue": f"""Create a dialogue scene between characters in a {genre} story. 
                Make the conversation natural and revealing.""",
                
                "Describe Setting": f"""Paint a vivid picture of a setting for a {genre} story using 
                {writing_style} style. Create atmosphere and mood.""",
                
                "Write Complete Short Story": f"""Write a complete short story in the {genre} genre with 
                approximately {word_limit} words using a {writing_style} style."""
            }
            
            base_prompt = prompts[assistance_type]
            user_prompt = f"{base_prompt}\nAdditional requirements: {additional_prompt}" if additional_prompt else base_prompt
            
            # Prepare the messages for the chat API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Prepare the request payload
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": False
            }
            
            # Send request to Ollama API
            response = requests.post(OLLAMA_API, json=payload, headers=HEADERS)
            
            if response.status_code == 200:
                generated_content = response.json()["message"]["content"]
                return generated_content
            else:
                return f"Error: API returned status code {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Please make sure Ollama is running on localhost:11434"
        except Exception as e:
            return f"Error: {str(e)}"

def create_interface():
    assistant = StoryAssistant()
    
    with gr.Blocks() as interface:
        gr.Markdown("# Creative Writing Assistant")
        
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=assistant.models,
                label="Select Model",
                value="mistral:latest"
            )
            assistance_type = gr.Dropdown(
                choices=assistant.assistance_types,
                label="What kind of help do you need?",
                value="Generate Story Idea"
            )
        
        with gr.Row():
            genre_dropdown = gr.Dropdown(
                choices=assistant.genres,
                label="Genre",
                value="Fantasy"
            )
            writing_style = gr.Dropdown(
                choices=assistant.writing_styles,
                label="Writing Style",
                value="Descriptive"
            )
        
        with gr.Row():
            tone = gr.Radio(
                choices=["Light and Uplifting", "Neutral", "Dark and Serious"],
                label="Tone",
                value="Neutral"
            )
            word_limit = gr.Slider(
                minimum=100,
                maximum=2000,
                value=500,
                step=100,
                label="Approximate Word Limit"
            )
        
        additional_prompt = gr.Textbox(
            lines=3,
            label="Additional Requirements (Optional)",
            placeholder="Add any specific elements, themes, or requirements you'd like to include..."
        )
        
        generate_button = gr.Button("Generate")
        
        output_text = gr.Textbox(
            lines=12,
            label="Generated Content",
            show_copy_button=True
        )
        
        # Add example prompts for different assistance types
        gr.Examples(
            examples=[
                ["Generate a story about a time traveler who can only travel to random times"],
                ["Create a character who is a retired superhero trying to live a normal life"],
                ["Describe a futuristic city where dreams are bought and sold"],
                ["Write a dialogue between two AI programs falling in love"],
            ],
            inputs=additional_prompt,
            label="Example Prompts"
        )
        
        generate_button.click(
            fn=assistant.generate_content,
            inputs=[
                model_dropdown,
                assistance_type,
                genre_dropdown,
                writing_style,
                additional_prompt,
                word_limit,
                tone
            ],
            outputs=output_text
        )
    
    return interface

# Create and launch the interface
interface = create_interface()
interface.launch(share=False)
