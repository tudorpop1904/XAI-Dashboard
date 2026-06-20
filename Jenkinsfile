pipeline {
    agent any

    triggers {
        pollSCM('H/5 * * * *')
        githubPush()
    }

    environment {
        PIP_CACHE_DIR = "${JENKINS_HOME}/.cache/pip"
    }

    stages {

        /* ---------------- CI: LINT ---------------- */
        stage('Lint') {
            steps {
                sh '''
                    python3 -m pip install --upgrade pip
                    pip install ruff
                    ruff check . --output-format=github
                '''
            }
        }

        /* ---------------- CI: TEST ---------------- */
        stage('Unit Tests') {
            steps {
                sh '''
                    python3 -m pip install --upgrade pip
                    pip install pytest pytest-cov
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

        /* ---------------- DEPLOY ---------------- */
        stage('Deploy') {
            when {
                branch 'main'
            }

            steps {
                sh '''
                    docker compose -f docker/docker-compose.yml up -d --remove-orphans
                '''
            }
        }
    }
}