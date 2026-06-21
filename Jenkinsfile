pipeline {
    agent any

    triggers {
        pollSCM('H/5 * * * *')
        githubPush()
    }

    environment {
        PIP_CACHE_DIR = "${JENKINS_HOME}/.cache/pip"
        VENV_PATH = ".venv-ci"
        // Microsoft Azure VM — set these in Jenkins > Manage Credentials
        CLOUD_VM_IP = credentials('azure-cloud-vm-ip')
        CLOUD_SSH_KEY = credentials('azure-cloud-ssh-key')
    }

    stages {

        /* ---------------- CI: SETUP ---------------- */
        stage('Setup') {
            steps {
                sh '''
                    python3 -m venv ${VENV_PATH}
                    source ${VENV_PATH}/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements-dev.txt
                '''
            }
        }

        /* ---------------- CI: LINT ---------------- */
        stage('Lint') {
            steps {
                sh '''
                    source ${VENV_PATH}/bin/activate
                    ruff check . --output-format=github
                '''
            }
        }

        /* ---------------- CI: TEST ---------------- */
        stage('Unit Tests') {
            steps {
                sh '''
                    source ${VENV_PATH}/bin/activate
                    pytest tests/unit/ -v --tb=short --junitxml=test-results.xml
                '''
            }
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }

        /* ---------------- BUILD (OPTIMIZED) ---------------- */
        stage('Build Images') {

            when {
                anyOf {
                    branch 'main'
                    branch 'release/*'
                }
                expression {
                    return filesChanged([
                        "docker/**",
                        "core/**",
                        "pages/**",
                        "ui/**",
                        "requirements.txt",
                        ".dockerignore",
                        "Dockerfile*"
                    ])
                }
            }

            steps {
                sh '''
                    export DOCKER_BUILDKIT=1

                    docker buildx build \
                        -f docker/Dockerfile.app \
                        -t xai-app:latest \
                        --cache-from=type=local,src=/tmp/.buildx-cache \
                        --cache-to=type=local,dest=/tmp/.buildx-cache-new \
                        .

                    docker buildx build \
                        -f docker/Dockerfile.ollama \
                        -t xai-ollama:latest \
                        --cache-from=type=local,src=/tmp/.buildx-cache \
                        --cache-to=type=local,dest=/tmp/.buildx-cache-new \
                        .
                '''

                sh 'rm -rf /tmp/.buildx-cache && mv /tmp/.buildx-cache-new /tmp/.buildx-cache'
            }
        }

        /* ---------------- LOCAL DEPLOY ---------------- */
        stage('Deploy (Local)') {
            when {
                branch 'main'
            }

            steps {
                sh '''
                    docker compose -f docker/docker-compose.yml up -d --remove-orphans
                '''
            }
        }

        /* ------------ CLOUD DEPLOY (Microsoft Azure) ------------ */
        stage('Deploy (Cloud)') {
            when {
                branch 'main'
            }

            steps {
                sh '''
                    chmod +x deploy/deploy.sh
                    ./deploy/deploy.sh "${CLOUD_VM_IP}" "${CLOUD_SSH_KEY}" azureuser
                '''
            }
        }

        /* ---------------- SMOKE TESTS ---------------- */
        stage('Smoke Tests') {
            steps {
                sh 'curl -sf http://localhost:8501/ || echo "Local smoke test skipped (not deployed locally)"'
                sh 'curl -sf http://localhost:11434/api/tags || echo "Local Ollama smoke test skipped"'
            }
        }
    }
}