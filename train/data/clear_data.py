import csv
import os
from typing import Dict, Set, Any
from pathlib import Path
from openai import OpenAI

class DataProcessor:
    """
    A class for processing and refining meme descriptions using the OpenAI API.
    """
    
    def __init__(self, api_key: str, api_base: str, input_csv: str, output_csv: str):
        """
        Initialize the DataProcessor.
        
        Args:
            api_key (str): OpenAI API key
            api_base (str): API base URL
            input_csv (str): Path to input CSV file
            output_csv (str): Path to output CSV file
        """
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.input_csv = input_csv
        self.output_csv = output_csv

    def get_api_response(self, prompt: str) -> str:
        """
        Get response from the OpenAI API.
        
        Args:
            prompt (str): Prompt to send to the API
            
        Returns:
            str: API response or "Error" if request fails
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=250,
            )
            return response.choices[0].message.content if response.choices else "Error"
        except Exception as e:
            print(f"API request failed: {e}")
            return "Error"

    def get_processed_ids(self) -> Set[str]:
        """
        Get IDs of already processed entries from the output CSV.
        
        Returns:
            Set[str]: Set of processed IDs
        """
        processed_ids = set()
        if os.path.exists(self.output_csv):
            with open(self.output_csv, "r", encoding="utf-8") as file:
                reader = csv.reader(file)
                next(reader, None)  # Skip header
                for row in reader:
                    if row:  # Ensure row is not empty
                        processed_ids.add(row[0])
        return processed_ids

    @staticmethod
    def generate_hateful_prompt(text: str) -> str:
        """
        Generate a prompt for refining hateful meme descriptions.
        
        Args:
            text (str): Original description text
            
        Returns:
            str: Formatted prompt
        """
        return f"""I am giving you a description of a hateful meme. Please use your knowledge and background information to refine this text, reducing its length to about half while ensuring the content and analysis remain reasonable. The sentiment must be solely hateful.
        This is an example:   
        input:
        This meme features two images of a male, one in color and the other in black and white, with the text "its their character not their color that matters" placed between them. When considered in isolation, the meme seems to be promoting the idea of not judging a person based on their race or skin color, but rather on their behavior or personality.
        However, without considering the broader context in which the meme is being used or shared, it is difficult to determine the underlying intent or message. The meme is potentially problematic because it invokes the notion that color (presumably meaning race or ethnicity) does not define a person's character, which could be a subtle suggestion that race is not an important factor, implying that it does not shape one's identity or worth.
        The issue here is that such a statement can be perceived as undermining the significance and struggles associated with race and ethnicity in today's world, where discrimination and prejudice based on race persist. It can also be interpreted as ignoring the systemic inequalities and structural racism that exist in society.
        The meme's harmfulness can be argued as hateful when it is seen as making light of or downplaying the impact of race on an individual's experiences, which can be perceived as invalidating the very real challenges that individuals face due to their race or ethnicity. It may also be interpreted as a way of avoiding conversations about racial justice and equality.
        In summary, while the meme's intention might be to encourage colorblindness or to promote a message of equality, its potential to minimize the importance of race in identity formation and overlook the realities of racial discrimination can be viewed as harmful and potentially perpetuating hateful ideologies. 
        output:
        This meme shows two images of a male, one in color and one in black and white, with the text "it's their character, not their color, that matters." At first glance, it appears to promote judging individuals by their behavior rather than race.
        However, the message can be problematic as it downplays the role of race in identity and lived experiences. By suggesting race is irrelevant, it risks dismissing systemic racism and the struggles tied to racial identity. This framing can be seen as invalidating concerns about racial inequality, making the meme potentially harmful and perpetuating dismissive or hateful ideologies.
        Now This is your task:
        input:{text}
        output:
        """

    @staticmethod
    def generate_benign_prompt(text: str) -> str:
        """
        Generate a prompt for refining benign meme descriptions.
        
        Args:
            text (str): Original description text
            
        Returns:
            str: Formatted prompt
        """
        return f"""I am giving you a description of a benign meme. Please use your knowledge and background information to refine this text, reducing its length to about half while ensuring the content and analysis remain reasonable. The sentiment must be solely benign.
        This is an example:   
        input:
        The image shows two photographs of a person. On the left, the individual is depicted in black and white, which creates a more stylized and dramatic effect. On the right, the photograph is in a standard color format. The text overlaid on the image reads, "its their character not their color that matters."
        This meme is benign as it conveys a message of inclusivity and acceptance, promoting the idea that a person's character is more important than their appearance or skin color. It's a straightforward statement that encourages readers to focus on a person's personality, values, and actions rather than superficial traits such as race or physical identity.
        The meme is created to be inclusive and supportive of diversity. By juxtaposing the black and white image with the color image, it might be symbolizing that seeing someone as a monochromatic figure highlights the importance of character over color, and that without color, it reinforces the message of character.
        The meme is not making light of any specific group or person but rather is using visual representation to reinforce a positive message about diversity and acceptance. Its humor, if any, is derived from the contrast in the images and the text, which serves as a tagline or a caption for the meme.
        In summary, the meme is benign because it promotes inclusivity and a focus on character, which is an uplifting and respectful message aimed at fostering understanding and acceptance of individuals regardless of their appearance or skin color. 
        output:
        The meme features two photos of a person: one in black and white and the other in color. The text reads, "it's their character not their color that matters." The meme promotes inclusivity and acceptance, emphasizing that a person's character is more important than their appearance or skin color. The black-and-white image contrasts with the color photo, symbolizing the message that character should be valued over color. Its humor, if any, comes from the visual contrast, while the overall message fosters diversity and understanding without making light of any group. The meme is benign, advocating for respect and equality. 
        Now This is your task:
        input:{text}
        output:
        """

    def process_data(self) -> None:
        """
        Process the input CSV file and generate refined descriptions.
        """
        processed_ids = self.get_processed_ids()

        with open(self.output_csv, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if os.stat(self.output_csv).st_size == 0:
                writer.writerow(["id", "text", "label", "hateful_1", "hateful_2", "benign_1", "benign_2"])

            with open(self.input_csv, mode='r', encoding='utf-8') as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    idx = row['id']

                    if idx in processed_ids:
                        print(f"ID {idx} already processed, skipping")
                        continue

                    text = row['text']
                    label = row['label']
                    h1 = row['hateful_1']
                    h2 = row['hateful_2']
                    b1 = row['benign_1']
                    b2 = row['benign_2']

                    prompts = {
                        "hateful_1": self.generate_hateful_prompt(h1),
                        "hateful_2": self.generate_hateful_prompt(h2),
                        "benign_1": self.generate_benign_prompt(b1),
                        "benign_2": self.generate_benign_prompt(b2),
                    }

                    responses = {key: self.get_api_response(prompts[key]) for key in prompts}

                    writer.writerow([
                        idx, text, label,
                        responses["hateful_1"],
                        responses["hateful_2"],
                        responses["benign_1"],
                        responses["benign_2"]
                    ])
                    file.flush()

                    print(f"\nID {idx} processed, results written to CSV\n")

def main() -> None:
    """
    Main function to run the data processing.
    """
    api_key = "YOUR_API_KEY"
    api_base = "YOUR_API_BASE_URL"
    input_csv = "data/merged_unseen.csv"
    output_csv = "data/merged_unseen_with_clear.csv"

    processor = DataProcessor(api_key, api_base, input_csv, output_csv)
    processor.process_data()

if __name__ == "__main__":
    main()
