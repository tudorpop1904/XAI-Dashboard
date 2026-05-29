pipeline {
    agent any

    triggers {
        // Automatically check for new commits every 5 minutes
        pollSCM('H/5 * * * *')
    }

    environment {
        VENV = '.venv-ci'
        // Persist pip cache across builds so we don't redownload Torch
        PIP_CACHE_DIR = "${JENKINS_HOME}/.cache/pip"
    }

    stages {
        stage('Setup') {
            steps {
                sh '''
                    # Only create venv if it doesn't exist to save time
                    if [ ! -d "${VENV}" ]; then
                        python3 -m venv ${VENV}
                    fi
                    . ${VENV}/bin/activate
                    pip install --upgrade pip
                    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
                    pip install -r requirements-dev.txt
                '''
            }
        }

        stage('Lint') {
            steps {
                sh '''
                    . ${VENV}/bin/activate
                    ruff check --select I --fix
                    ruff format
                    ruff check . --output-format=github
                '''
            }
        }

        stage('Unit Tests') {
            steps {
                sh '''
                    . ${VENV}/bin/activate
                    pytest tests/unit/ -v --tb=short --junitxml=test-results.xml
                '''
            }
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }

        stage('Docker Build') {
            steps {
                sh 'docker build -f docker/Dockerfile.app -t xai-app:${BUILD_NUMBER} -t xai-app:latest .'
            }
        }

        stage('Deploy') {
            steps {
                echo 'Deploying application stack to the local Docker daemon...'
                // Start/recreate all services in detached mode
                sh 'docker compose -f docker/docker-compose.yml up -d'
            }
        }
    }

    // Removed cleanWs() so the .venv-ci directory persists across builds
    // making subsequent runs MUCH faster.
}
