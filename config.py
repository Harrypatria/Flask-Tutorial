MODEL_CONFIG = {
    # Path configurations
    'OUTPUT_DIR': 'model_artifacts',
    'MODEL_PATH': 'model_artifacts/model.pkl',
    'METADATA_PATH': 'model_artifacts/metadata.pkl',
    
    # Model configurations
    'VERSION': '1.0.1',
    'FEATURE_COLUMNS': ['R&D_Spend', 'Administration', 'Marketing_Spend', 'State'],
    'TARGET_COLUMN': 'Profit',
    'STATE_ENCODING': {
        'New York': 0,
        'California': 1,
        'Florida': 2
    },
    
    # Training configurations
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': 42
}
