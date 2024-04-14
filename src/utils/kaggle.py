import subprocess

from dotenv import load_dotenv

load_dotenv()


def submit_to_kaggle(competition_name, file_name, message):
    args = f'{competition_name} -f {file_name} -m "{message}"'
    command = "kaggle competitions submit -c" + args
    try:
        submission_response = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print(submission_response.stdout)
        return submission_response.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error in submission: {e.stderr}")
        return None


def check_submission_status(competition_name):
    command = f"kaggle competitions submissions {competition_name} --csv"
    try:
        submission_status = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        #        print(submission_status.stdout)
        return submission_status.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error checking submission status: {e.stderr}")
        return None
