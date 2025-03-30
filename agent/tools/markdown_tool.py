import os
import re
from typing import Optional, Union, Dict, Any
from langchain.llms import BaseLLM
from langchain.tools import Tool
from pydantic import BaseModel, Field

# Markdown rendering and PDF conversion dependencies
import markdown
import pdfkit
import pypandoc


class MarkdownWriterTool:
    """
    A tool that uses LLM to generate markdown content.
    """

    def __init__(self, llm: BaseLLM):
        """
        Initialize the Markdown Writing Tool with an LLM.

        Args:
            llm (BaseLLM): Language model for content generation
        """
        self.llm = llm

    def generate_markdown(self, topic: str, content_type: str,
                          additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a well-structured markdown document using LLM.

        Args:
            topic (str): Main topic of the document
            content_type (str): Type of document (e.g., 'blog_post', 'research_paper', 'technical_guide')
            additional_context (dict, optional): Additional context or requirements

        Returns:
            str: Structured markdown content
        """
        # Prepare prompt for LLM
        context_str = f"Additional context: {additional_context}" if additional_context else ""
        prompt = f"""
        Create a comprehensive, well-structured markdown document on the topic: "{topic}"

        Document Type: {content_type}
        {context_str}

        Guidelines:
        - Use markdown formatting extensively
        - Include clear and descriptive headings
        - Organize content with logical flow
        - Use bullet points, numbered lists, and emphasis
        - Include relevant sections like introduction, key points, conclusion
        - Add markdown-specific formatting like code blocks, tables, or blockquotes if appropriate
        - Cite sources or add references if applicable

        Provide the complete markdown document.
        """

        try:
            # Generate markdown using LLM
            markdown_content = self.llm.generate(prompt)

            # Clean and format the markdown
            return self._clean_markdown(markdown_content)
        except Exception as e:
            return f"Error generating markdown: {str(e)}"

    def _clean_markdown(self, content: str) -> str:
        """
        Clean and standardize the generated markdown.

        Args:
            content (str): Raw markdown content

        Returns:
            str: Cleaned markdown content
        """
        # Remove any code block markers or LLM artifacts
        content = re.sub(r'```markdown\n?|```\n?', '', content)

        # Normalize headings
        content = re.sub(r'^#+\s*', lambda m: '#' * len(m.group(0).strip()) + ' ', content, flags=re.MULTILINE)

        # Ensure proper spacing
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content.strip()


class FileSaverTool:
    """
    Advanced file saving tool with markdown and PDF rendering capabilities.
    """

    def __init__(self, base_path: str = "./saved_files"):
        """
        Initialize the File Saver Tool.

        Args:
            base_path (str, optional): Base directory for saving files
        """
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def save_markdown_file(self, markdown_content: str, filename: str) -> str:
        """
        Save markdown content to a file.

        Args:
            markdown_content (str): Markdown content to save
            filename (str): Name of the file

        Returns:
            str: Confirmation message
        """
        # Ensure .md extension
        if not filename.endswith('.md'):
            filename += '.md'

        full_path = os.path.join(self.base_path, filename)

        try:
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            return f"Markdown file saved: {filename}"
        except Exception as e:
            return f"Error saving markdown file: {str(e)}"

    def save_markdown_as_pdf(self, markdown_content: str, filename: str) -> str:
        """
        Convert markdown to PDF using multiple methods.

        Args:
            markdown_content (str): Markdown content to convert
            filename (str): Output filename

        Returns:
            str: Confirmation message
        """
        # Ensure .pdf extension
        if not filename.endswith('.pdf'):
            filename += '.pdf'

        full_path = os.path.join(self.base_path, filename)

        try:
            # Try multiple conversion methods
            methods = [
                self._convert_with_pandoc,
                self._convert_with_pdfkit,
                self._convert_with_basic_html
            ]

            for method in methods:
                try:
                    method(markdown_content, full_path)
                    return f"PDF file saved: {filename}"
                except Exception as inner_e:
                    print(f"Conversion method failed: {str(inner_e)}")

            raise Exception("All PDF conversion methods failed")

        except Exception as e:
            return f"Error saving PDF: {str(e)}"

    def _convert_with_pandoc(self, markdown_content: str, output_path: str):
        """
        Convert markdown to PDF using Pandoc.

        Args:
            markdown_content (str): Markdown content
            output_path (str): Path to save PDF
        """
        # Requires pandoc to be installed
        pypandoc.convert_text(
            markdown_content,
            'pdf',
            format='md',
            outputfile=output_path,
            extra_args=['-V', 'geometry:margin=1in']
        )

    def _convert_with_pdfkit(self, markdown_content: str, output_path: str):
        """
        Convert markdown to PDF using pdfkit (requires wkhtmltopdf).

        Args:
            markdown_content (str): Markdown content
            output_path (str): Path to save PDF
        """
        # Convert markdown to HTML first
        html_content = markdown.markdown(markdown_content)

        # Wrap in basic HTML structure
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Convert to PDF
        pdfkit.from_string(full_html, output_path)

    def _convert_with_basic_html(self, markdown_content: str, output_path: str):
        """
        Fallback method: Convert markdown to basic PDF via HTML.

        Args:
            markdown_content (str): Markdown content
            output_path (str): Path to save PDF
        """
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content)

        # Use basic HTML to PDF conversion
        with open(output_path.replace('.pdf', '.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)

        raise Exception("Basic HTML conversion (not true PDF)")


# Input schemas for Langchain Tool compatibility
class MarkdownGenerateInput(BaseModel):
    topic: str = Field(..., description="Main topic of the document")
    content_type: str = Field(..., description="Type of document (e.g., 'blog_post', 'research_paper')")
    additional_context: Optional[Dict[str, Any]] = Field(None, description="Additional context or requirements")


# Function to create Langchain tools
def create_markdown_tools(llm: BaseLLM):
    """
    Create Langchain tools for markdown writing and saving.

    Args:
        llm (BaseLLM): Language model to use for content generation

    Returns:
        list: Langchain tools
    """
    markdown_writer = MarkdownWriterTool(llm)
    file_saver = FileSaverTool()

    return [
        Tool(
            name="generate_markdown",
            func=markdown_writer.generate_markdown,
            description="Generate a structured markdown document using LLM. **Required:** topic (str), content_type (str), additional_context (optional)",
            args_schema=MarkdownGenerateInput
        ),
        Tool(
            name="save_markdown_file",
            func=file_saver.save_markdown_file,
            description="Save markdown content to a file. **Required:** markdown_content (str), filename (str)"
        ),
        Tool(
            name="save_markdown_as_pdf",
            func=file_saver.save_markdown_as_pdf,
            description="Convert markdown to PDF. **Required:** markdown_content (str), filename (str)"
        )
    ]