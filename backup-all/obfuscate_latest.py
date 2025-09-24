#!/usr/bin/env python3
"""
Safe PyArmor obfuscation script for newstartup.py
This script ensures compatibility with existing obfuscated files
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_pyarmor():
    """Check if PyArmor is available."""
    try:
        result = subprocess.run(['pyarmor', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ PyArmor found: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ PyArmor not found. Install with: pip install pyarmor==9.1.8")
        return False

def check_existing_runtime():
    """Check for existing PyArmor runtime in the directory."""
    runtime_dirs = [
        'pyarmor_runtime_000000',
        'pyarmor_runtime_000001', 
        'pyarmor_runtime_000002',
        'pyarmor_runtime_000003'
    ]
    
    existing_runtimes = []
    for runtime_dir in runtime_dirs:
        if Path(runtime_dir).exists():
            existing_runtimes.append(runtime_dir)
    
    if existing_runtimes:
        print(f"✅ Found existing runtime directories: {existing_runtimes}")
        return existing_runtimes[0]  # Use the first one found
    else:
        print("⚠️  No existing runtime directories found")
        return None

def create_custom_runtime(runtime_name="xoxoxo"):
    """Create a custom-named runtime directory for newstartup.py."""
    print(f"🔄 Creating custom runtime directory: {runtime_name}")
    
    # For PyArmor, we need to use a different approach
    # Instead of creating a custom runtime, we'll use the existing one
    # and just rename the output directory
    existing_runtime = check_existing_runtime()
    if existing_runtime:
        print(f"✅ Using existing runtime: {existing_runtime}")
        return existing_runtime
    
    # If no existing runtime, create one with default name
    print("🔄 No existing runtime found, will create default one")
    return None

def obfuscate_newstartup_safe(use_custom_runtime=False):
    """Obfuscate newstartup.py safely without conflicting with existing files."""
    
    if not check_pyarmor():
        return False
    
    # Check if newstartup.py exists
    if not Path('newstartup.py').exists():
        print("❌ newstartup.py not found in current directory")
        return False
    
    # Check for existing runtime
    existing_runtime = check_existing_runtime()
    
    # Create backup
    if Path('newstartup.py').exists():
        shutil.copy2('newstartup.py', 'newstartup.py.backup')
        print("✅ Created backup: newstartup.py.backup")
    
    # Method 1: Try to use existing runtime directory (unless custom runtime is requested)
    if existing_runtime and not use_custom_runtime:
        print(f"🔄 Attempting to use existing runtime: {existing_runtime}")
        
        # Copy newstartup.py to a temp location
        temp_dir = Path('temp_obfuscation')
        temp_dir.mkdir(exist_ok=True)
        
        try:
            # Try to obfuscate using the same runtime
            cmd = [
                'pyarmor', 'gen', 
                '--obf-code', '0',  # Minimal obfuscation for compatibility
                '--obf-module', '0',
                '--use-runtime', existing_runtime,  # Use existing runtime
                '-O', str(temp_dir),
                'newstartup.py'
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Obfuscation successful with existing runtime")
                
                # Copy the obfuscated file back
                obfuscated_file = temp_dir / 'newstartup.py'
                if obfuscated_file.exists():
                    shutil.copy2(obfuscated_file, 'newstartup.py')
                    print("✅ Replaced newstartup.py with obfuscated version")
                    
                    # Copy any new runtime files to existing runtime directory
                    for runtime_candidate in ['pyarmor_runtime_000000', 'pyarmor_runtime_000001']:
                        new_runtime = temp_dir / runtime_candidate
                        if new_runtime.exists() and new_runtime != Path(existing_runtime):
                            print(f"⚠️  New runtime created: {runtime_candidate}")
                            print("   This might cause conflicts with existing obfuscated files")
                    
                    return True
                else:
                    print("❌ Obfuscated file not found in output")
                    return False
            else:
                print(f"❌ Obfuscation failed: {result.stderr}")
                return False
                
        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    # Method 2: Create new obfuscation and rename runtime directory
    print("🔄 Creating new obfuscation with custom runtime directory")
    
    cmd = [
        'pyarmor', 'gen',
        '--obf-code', '0',  # Minimal obfuscation
        '--obf-module', '0',
        'newstartup.py'
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Obfuscation successful")
        
        # Check if files were created in dist/ directory
        dist_dir = Path("dist")
        if dist_dir.exists():
            print("✅ Found obfuscated files in dist/ directory")
            
            # Move obfuscated file from dist/ to current directory
            obfuscated_file = dist_dir / "newstartup.py"
            if obfuscated_file.exists():
                shutil.copy2(obfuscated_file, "newstartup.py")
                print("✅ Moved obfuscated newstartup.py to current directory")
            
            # Find and rename runtime directory
            runtime_dirs = list(dist_dir.glob("pyarmor_runtime_*"))
            if runtime_dirs:
                runtime_dir = runtime_dirs[0]
                custom_runtime_name = "xoxoxo"
                
                # Remove existing custom runtime if it exists
                if Path(custom_runtime_name).exists():
                    shutil.rmtree(custom_runtime_name)
                
                # Move and rename the runtime directory
                shutil.move(str(runtime_dir), custom_runtime_name)
                print(f"✅ Renamed runtime directory to: {custom_runtime_name}")
            
            # Clean up dist directory
            shutil.rmtree(dist_dir)
            print("✅ Cleaned up dist/ directory")
        
        return True
    else:
        print(f"❌ Obfuscation failed: {result.stderr}")
        return False

def test_obfuscated_file():
    """Test if the obfuscated newstartup.py works."""
    print("🧪 Testing obfuscated newstartup.py...")
    
    try:
        # Test import and basic functionality
        result = subprocess.run([
            'python3', '-c', 
            '''
import sys
sys.path.insert(0, ".")
try:
    import newstartup
    print("✅ Import successful")
    
    # Test if main functions exist
    if hasattr(newstartup, "setup_authentication"):
        print("✅ setup_authentication function found")
    if hasattr(newstartup, "main"):
        print("✅ main function found")
    
    print("✅ All tests passed")
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
            '''
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Obfuscated newstartup.py works correctly")
            return True
        else:
            print(f"❌ Test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main function."""
    print("🔧 Safe PyArmor Obfuscation for newstartup.py")
    print("=" * 50)
    
    # Check for help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        print("Usage: python obfuscate.py [--custom-runtime]")
        print("")
        print("Options:")
        print("  --custom-runtime    Create a custom runtime directory (xoxoxo)")
        print("                     instead of using existing pyarmor_runtime_* directories")
        print("")
        print("Default behavior: Try to use existing runtime first, then create custom if needed")
        sys.exit(0)
    
    # Check current directory
    if not Path('newstartup.py').exists():
        print("❌ Please run this script from the directory containing newstartup.py")
        sys.exit(1)
    
    # Check command line arguments
    use_custom_runtime = "--custom-runtime" in sys.argv
    if use_custom_runtime:
        print("🎯 Using custom runtime directory for newstartup.py")
    else:
        print("🎯 Will try to use existing runtime first, then create custom if needed")
    
    # Obfuscate the file
    if obfuscate_newstartup_safe(use_custom_runtime):
        print("\n✅ Obfuscation completed successfully")
        
        # Test the obfuscated file
        if test_obfuscated_file():
            print("\n🎉 All done! newstartup.py is obfuscated and working")
        else:
            print("\n⚠️  Obfuscation completed but testing failed")
            print("   You may need to restore from backup: mv newstartup.py.backup newstartup.py")
    else:
        print("\n❌ Obfuscation failed")
        print("   Restore from backup: mv newstartup.py.backup newstartup.py")
        sys.exit(1)

if __name__ == "__main__":
    main()