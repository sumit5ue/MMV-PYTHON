import boto3

def get_rekognition_client():
    # Initialize the Rekognition client
    rekognition_client = boto3.client('rekognition', region_name='us-west-2')  # Replace with your region
    return rekognition_client
