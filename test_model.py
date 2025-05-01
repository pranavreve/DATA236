import os
import sys
import joblib

def test_model_exists():
    """Test if the model file exists."""
    assert os.path.exists('model.pkl'), "Model file does not exist"
    print("✅ Model file exists")

def test_model_loads():
    """Test if the model can be loaded."""
    try:
        model = joblib.load('model.pkl')
        print("✅ Model loads successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def test_feature_names_exist():
    """Test if the feature names file exists."""
    assert os.path.exists('feature_names.pkl'), "Feature names file does not exist"
    print("✅ Feature names file exists")

if __name__ == "__main__":
    tests_passed = True
    
    try:
        test_model_exists()
    except AssertionError as e:
        print(f"❌ {e}")
        tests_passed = False
    
    try:
        model_loads = test_model_loads()
        if not model_loads:
            tests_passed = False
    except Exception as e:
        print(f"❌ Error during model loading test: {e}")
        tests_passed = False
    
    try:
        test_feature_names_exist()
    except AssertionError as e:
        print(f"❌ {e}")
        tests_passed = False
    
    if tests_passed:
        print("All tests passed! ✅")
        sys.exit(0)
    else:
        print("Some tests failed! ❌")
        sys.exit(1) 