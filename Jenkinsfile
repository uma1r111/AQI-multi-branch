pipeline {
    agent any

    triggers {
        cron('H 6 * * *')  // Runs daily at 11 PM PKT
    }

    environment {
        PYTHONPATH = "${env.WORKSPACE}"
        VENV_PATH = "${env.WORKSPACE}/.venv"
    }

    stages {
        // 🔲 Dev Placeholder
        stage('Dev Logic Placeholder') {
            when {
                branch 'dev'
            }
            steps {
                echo "🚧 Dev logic is not active in post-production branch."
            }
        }

        // 🔲 Pre-Production Placeholder
        stage('Pre-Production Logic Placeholder') {
            when {
                branch 'pre-production'
            }
            steps {
                echo "🚧 Pre-production logic is not active in post-production branch."
            }
        }

        // ✅ Post-Production Logic
        stage('Create Virtual Environment') {
            steps {
                echo "Creating virtual environment at ${VENV_PATH}..."
                sh 'python3 -m venv .venv'
            }
        }

        stage('Install Dependencies') {
            steps {
                echo "Installing Python dependencies..."
                sh '''
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Run Post-Production Deployment') {
            steps {
                echo "🚀 Running post_prod_deploy.py from root directory..."
                sh '''
                    . .venv/bin/activate
                    python "post_prod_deploy.py"
                '''
            }
        }
    }

    post {
        success {
            echo '✅ Post-production pipeline completed successfully.'
        }
        failure {
            echo '❌ Post-production pipeline failed. Please check error logs.'
        }
    }
}
