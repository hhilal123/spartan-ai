import datetime
import re
import time
from html.parser import HTMLParser


#############################################################
# CustomError handler
#############################################################
class CustomError(Exception):
    """Custom exception class."""

    def __init__(self, message, error_type):
        self.message = message
        self.error_type = error_type
        super().__init__(self.message)


#############################################################
# Create a custom HTML parser that inherits from HTMLParser
#############################################################


class NSHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = ""

    def handle_data(self, data):
        self.result += data


class UtilFunctions:

    def __init__(self):
        return None

    #############################################################
    # Remove duplicate text from resoponse
    #############################################################

    def add_separator_to_duplicate_text(
        self, input_text, separator="::split_response_here::"
    ):
        parts = input_text.split("\n")
        add_split_message = False

        for i in range(1, len(parts)):
            if parts[i] == parts[0]:
                # check if all else matches as well
                if not add_split_message:
                    parts.insert(i, separator)
                    add_split_message = True

        return "\n".join(parts)

    #################################################
    # Get items with HTML
    #################################################
    def get_items_with_html(self, lst):
        pattern = r"<[^>]*>"
        items_with_html = []
        for item in lst:
            if re.search(pattern, item):
                items_with_html.append(item)
                break
        return items_with_html

    #################################################
    # Check if string is valid
    #################################################
    def check_valid_string(self, string):
        pattern = r"\{[^}]+\}"  # r"{[\w\\/]+}"  # r"{\w+}"
        if re.search(pattern, string):
            return False
        return True

    #################################################
    # Checking if the string is junk or not on special characters
    #################################################
    def is_junk_string(self, string):
        # Check for special characters more than 3
        if len(re.findall(r"[^\w\s]", string)) > 7:
            return True
        # # Check for proper digits (length 1, 2, 3, or 10)
        # digits = re.findall(r'\b\d+\b', string)
        # for digit in digits:
        #     if len(digit) not in [1, 2, 3, 10]:
        #         return True

        return False

    def has_special_character(self, text):
        pattern = r'[!@#$%^&*(),.?":{}|/<>\\]'
        return bool(re.search(pattern, text))

    #################################################
    # Check non module support in prompt
    #################################################
    def check_relevance(self, request):
        strings_list = ["edit", "help", "check", "write", "review"]
        for s in strings_list:
            if s in request:
                return False
        return True

    def process_link_on_linenumber(self, value):
        match = re.search(r"\[(\d+)\]\((.*?)\)", value)
        if match:
            number = match.group(1)
            url = match.group(2)
            return f'<a target="_blank" href="{url}">{number}</a>'
        else:
            return value

    #################################################
    # TXT TO HTML Format for LINK
    #################################################
    def text_to_html_with_link(self, text):
        # regex pattern [text](link) -- google
        pattern = r"\[([^]]+)\]\(([^)]+)\)"

        matches = re.findall(pattern, text)
        for match in matches:
            text_to_replace = f"[{match[0]}]({match[1]})"
            html_link = f'<a href="{match[1]}" target="_blank">{match[0]}</a>'
            text = text.replace(text_to_replace, html_link)
        return text

    def calculate_elapsed_time(self, start_time):
        """
        Calculate and format the elapsed time between the given start time and the current time.
        :param start_time: The start time as a float representing seconds since the epoch.
        :return: A formatted string representing the elapsed time in the format 'hh:mm:ss.sss'.
        """
        end_time = time.time()
        elapsed_time = end_time - start_time

        elapsed_timedelta = datetime.timedelta(seconds=elapsed_time)

        # Extract hours, minutes, seconds, and milliseconds
        hours = elapsed_timedelta.seconds // 3600
        minutes = (elapsed_timedelta.seconds // 60) % 60
        seconds = elapsed_timedelta.seconds % 60
        milliseconds = elapsed_timedelta.microseconds // 1000

        # Format the elapsed time
        elapsed_time_formatted = (
            f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
        )

        return elapsed_time_formatted

    def normalize_string(self, string: str):
        """Return an all lowercase string with all whitespace stripped."""
        return string.lower().strip().replace(" ", "")

