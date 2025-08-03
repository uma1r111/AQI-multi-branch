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
        stage('Create Virtual Environment') {
            steps {
                echo "Creating virtual environment at ${VENV_PATH}..."
                sh 'python3 -m venv .venv'
            }
        }

        stage('Install Dependencies') {
            steps {
                echo "Installing Python dependencies from requirements.txt..."
                sh '''
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Step 1: Fetch Daily Data') {
            steps {
                echo "Running fetch_daily_data.py..."
                sh '''
                    . .venv/bin/activate
                    python collect data/fetch_daily_data.py
                '''
            }
        }

        stage('Step 2: Feature Engineering') {
            steps {
                echo "Running run_preprocessing.py..."
                sh '''
                    . .venv/bin/activate
                    python feature engineering/run_preprocessing.py
                '''
            }
        }

        stage('Step 3: Feature Selection') {
            steps {
                echo "Running feature_selection.py..."
                sh '''
                    . .venv/bin/activate
                    python feature selection/feature_selection.py
                '''
            }
        }

        stage('Preview Final Output') {
            steps {
                echo "Showing last few rows of final output CSV..."
                sh "tail -n 10 data/feature_selection.csv || true"
            }
        }

        stage('Step 4: Run Model(s) from Config') {
            steps {
                echo "Checking model_config.json..."
                script {
                    def config = readJSON file: 'model_config.json'

                    if (config.run_sarimax == true) {
                        echo "Running SARIMAX model..."
                        sh '''
                            . .venv/bin/activate
                            python model training/sarimax.py
                        '''
                    } else if (config.run_XGB == true) {
                        echo "Running XGBoost model..."
                        sh '''
                            . .venv/bin/activate
                            python model training/XGB.py
                        '''
                    } else {
                        error("❌ No valid model selected in model_config.json")
                    }
                }
            }
        }

        stage('Step 5: Compare with Previous Best') {
            steps {
                echo "Running compare_model.py..."
                sh '''
                    . .venv/bin/activate
                    python model training/compare_model.py
                '''
            }
        }

        // 🔶 Pre-Production Placeholder
        stage('Pre-Production Logic') {
            when {
                expression {
                    // Replace this with logic to detect pre-prod branch or alias later
                    return false
                }
            }
            steps {
                echo "🚧 Pre-production logic will go here..."
                // TODO: Add pre-production retraining and evaluation
            }
        }

        // 🔷 Post-Production Placeholder
        stage('Post-Production Logic') {
            when {
                expression {
                    // Replace this with logic to detect post-prod alias later
                    return false
                }
            }
            steps {
                echo "🚧 Post-production deployment logic will go here..."
                // TODO: Add deployment tag updates, monitoring etc.
            }
        }
    }

    post {
        success {
            echo '✅ Dev pipeline completed successfully.'
        }
        failure {
            echo '❌ Dev pipeline failed. Please check error logs.'
        }
    }
}
