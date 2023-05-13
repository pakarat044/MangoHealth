FROM public.ecr.aws/lambda/python:3.10.2023.04.04.12

# Install the function's dependencies using file requirements.txt
# from your project folder.

COPY model_Pre-train_MobileNetV2 ./model_Pre-train_MobileNetV2

COPY requirements.txt  .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.lambda_handler" ] 