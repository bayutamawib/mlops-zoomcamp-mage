# Start from Mage base image
FROM mageai/mageai:alpha

# Define build arguments and environment variables
ARG PROJECT_NAME=mlops
ARG MAGE_CODE_PATH=/app/mlops
ARG USER_CODE_PATH=${MAGE_CODE_PATH}/${PROJECT_NAME}

# Set working directory inside the container
WORKDIR ${MAGE_CODE_PATH}

# Copy code into container
COPY . ${MAGE_CODE_PATH}

# Set environment variable for user code path
ENV USER_CODE_PATH=${USER_CODE_PATH}
ENV PYTHONPATH="${PYTHONPATH}:${USER_CODE_PATH}"

# Install custom Python libraries and dependencies
RUN if [ -f "${USER_CODE_PATH}/requirements.txt" ]; then pip3 install -r ${USER_CODE_PATH}/requirements.txt; fi

# Uncomment these lines to install Terraform (optional)
# RUN apt-get update && \
#     apt-get install -y wget unzip && \
#     wget https://releases.hashicorp.com/terraform/1.8.3/terraform_1.8.3_linux_amd64.zip && \
#     unzip terraform_1.8.3_linux_amd64.zip -d /usr/local/bin/ && \
#     rm terraform_1.8.3_linux_amd64.zip

# Define the command to run your application
CMD ["/bin/sh", "-c", "${MAGE_CODE_PATH}/run_app.sh"]
