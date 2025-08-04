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
                echo "🚧 Dev logic is not active in pre-production branch."
            }
        }

        // ✅ Pre-Production Logic
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

        stage('Step 1: Download Feature Selection from S3') {
            steps {
                echo "Downloading feature_selection.csv from S3..."
                sh '''
                    . .venv/bin/activate
                    python "feature selection/download_feature_selection.py"
                '''
            }
        }

        stage('Step 2: Retrain Pre-Challenger Model') {
            steps {
                echo "Retraining SARIMAX model using previous best config..."
                sh '''
                    . .venv/bin/activate
                    python "model training/sarimax.py"
                '''
            }
        }

        stage('Step 3: Evaluate & Update Alias') {
            steps {
                echo "Comparing model and updating alias if improved..."
                sh '''
                    . .venv/bin/activate
                    python "model training/compare_model.py"
                '''
            }
        }

        // 🔲 Post-Production Placeholder
        stage('Post-Production Logic Placeholder') {
            when {
                branch 'post-production'
            }
            steps {
                echo "🚧 Post-production logic is not active in pre-production branch."
            }
        }
    }

    post {
        success {
            echo '✅ Pre-production pipeline completed successfully.'
        }
        failure {
            echo '❌ Pre-production pipeline failed. Please check error logs.'
        }
    }
}
