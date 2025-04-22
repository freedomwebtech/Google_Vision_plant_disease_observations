import os
import base64
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Set your Google API key
GOOGLE_API_KEY = "AIzaSyAHQh7TwV4ysqt0YTicZfxDLuAkbgvlmNo"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

class PlantImageAnalyzer:
    def __init__(self, image_folder):
        self.image_folder = image_folder
        self.gemini_model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

        # Create separate output files for English and Hindi
        self.output_filename_en = f"plant_disease_observations_en_{time.strftime('%Y-%m-%d')}.txt"
        self.output_filename_hi = f"plant_disease_observations_hi_{time.strftime('%Y-%m-%d')}.txt"

        for filename in [self.output_filename_en, self.output_filename_hi]:
            if not os.path.exists(filename):
                with open(filename, "w", encoding="utf-8") as f:
                    f.write("Timestamp | Image | Observation\n")
                    f.write("-" * 100 + "\n")

    def encode_image_to_base64(self, image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_image(self, image_path, language="english"):
        try:
            base64_image = self.encode_image_to_base64(image_path)

            prompt_text = (
                f"You are an expert agricultural assistant. "
                f"Analyze the image of this plant and provide the following in {language}:\n\n"
                "1. Name of the disease (if any)\n"
                "2. Description of symptoms\n"
                "3. Suggested treatment or medicine\n\n"
                "Format your response clearly with bullet points."
            )

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        "description": "Plant image"
                    }
                ]
            )

            response = self.gemini_model.invoke([message])
            return response.content.strip()

        except Exception as e:
            return f"Error: {e}"

    def process_image(self, image_path):
        image_name = os.path.basename(image_path)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Get both English and Hindi analysis
        result_en = self.analyze_image(image_path, language="English")
        result_hi = self.analyze_image(image_path, language="Hindi")

        print(f"✅ Processed {image_name}")

        # Save English result
        with open(self.output_filename_en, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} | {image_name} | {result_en}\n\n")

        # Save Hindi result
        with open(self.output_filename_hi, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} | {image_name} | {result_hi}\n\n")

    def run_analysis(self):
        image_files = [
            os.path.join(self.image_folder, f)
            for f in os.listdir(self.image_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        print(f"🔍 Found {len(image_files)} images. Starting analysis...")

        threads = []
        for image_path in image_files:
            thread = threading.Thread(target=self.process_image, args=(image_path,), daemon=True)
            thread.start()
            threads.append(thread)

        for t in threads:
            t.join()

        print(f"📄 All results saved to:\n  - {self.output_filename_en}\n  - {self.output_filename_hi}")

if __name__ == "__main__":
    image_folder = "images"  # Replace with your image folder
    analyzer = PlantImageAnalyzer(image_folder)
    analyzer.run_analysis()
