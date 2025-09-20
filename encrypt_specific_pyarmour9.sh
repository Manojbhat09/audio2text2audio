#!/bin/bash

# PyArmor 9.x Specific Files Encryption Script
# Encrypts server.py files using PyArmor 9.x with comprehensive error handling
# Author: AI Assistant
# Version: 3.0 - Rewritten based on PyArmor documentation

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Configuration
readonly PYARMOR_CMD="pyarmor"
readonly TARGET_FILES=("server.py")
readonly RUNTIME_DIR="pyarmor_runtime_000000"
readonly BACKUP_SUFFIX=".backup"
readonly SCRIPT_VERSION="3.0"

# Global variables
DRY_RUN=false
RESTORE_MODE=false
LIST_MODE=false
FORCE_MODE=false
TEST_MODE=false
VERBOSE=false

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_debug() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${CYAN}[DEBUG]${NC} $1"
    fi
}

# Function to check if PyArmor 9.x is installed and working
check_pyarmor() {
    print_info "Checking PyArmor installation..."
    
    # Activate conda environment
    print_debug "Activating conda environment..."
    if [[ -f "/home/mbhat/miniconda/bin/activate" ]]; then
        source /home/mbhat/miniconda/bin/activate
        conda activate optionscope
        print_debug "Conda environment activated"
    else
        print_warning "Conda activation script not found, trying direct path"
    fi
    
    if ! command -v "$PYARMOR_CMD" &> /dev/null; then
        print_error "PyArmor not found. Please install PyArmor 9.x first."
        print_info "Install with: pip install pyarmor==9.1.8"
        print_info "Or: pip install pyarmor>=9.0.0"
        exit 1
    fi

    local version_output
    if ! version_output=$($PYARMOR_CMD --version 2>&1); then
        print_error "Failed to get PyArmor version: $version_output"
        exit 1
    fi

    local version=$(echo "$version_output" | head -n1)
    print_success "Found PyArmor: $version"

    # Extract version number more reliably
    local version_number=""
    if [[ "$version" =~ ([0-9]+\.[0-9]+\.[0-9]+) ]]; then
        version_number="${BASH_REMATCH[1]}"
    elif [[ "$version" =~ ([0-9]+\.[0-9]+) ]]; then
        version_number="${BASH_REMATCH[1]}"
    elif [[ "$version" =~ ([0-9]+) ]]; then
        version_number="${BASH_REMATCH[1]}"
    fi

    # Check if it's version 9.x
    if [[ -n "$version_number" ]] && [[ "$version_number" =~ ^9\. ]]; then
        print_success "PyArmor version $version_number is compatible with this script"
    else
        print_warning "PyArmor version $version_number detected. This script is optimized for PyArmor 9.x"
        print_info "Current version: $version"
        print_info "Recommended: pip install pyarmor==9.1.8"
        
        if [[ "$version_number" =~ ^[0-7]\. ]]; then
            print_error "PyArmor version $version_number is too old. Please upgrade to 9.x"
            exit 1
        fi
    fi

    # Test PyArmor functionality
    print_debug "Testing PyArmor functionality..."
    if ! $PYARMOR_CMD --help &> /dev/null; then
        print_error "PyArmor is not functioning correctly"
        exit 1
    fi
}

# Function to check if directory exists and is accessible
check_directory() {
    local dir="$1"
    
    if [[ ! -d "$dir" ]]; then
        print_error "Directory does not exist: $dir"
        exit 1
    fi
    
    if [[ ! -r "$dir" ]] || [[ ! -w "$dir" ]]; then
        print_error "Directory is not accessible (read/write): $dir"
        exit 1
    fi
    
    print_success "Directory '$dir' is accessible"
}

# Function to check if target files exist
check_files() {
    local dir="$1"
    local missing_files=()
    local found_files=()

    for file in "${TARGET_FILES[@]}"; do
        if [[ ! -f "${dir}/${file}" ]]; then
            missing_files+=("$file")
        else
            found_files+=("$file")
        fi
    done

    if [[ ${#missing_files[@]} -gt 0 ]]; then
        print_warning "Missing files: ${missing_files[*]} (will be ignored)"
    fi

    if [[ ${#found_files[@]} -gt 0 ]]; then
        print_success "Found files to process: ${found_files[*]}"
    else
        print_error "No target files found in directory"
        exit 1
    fi
}

# Function to create backup
create_backup() {
    local file="$1"
    local backup_file="${file}${BACKUP_SUFFIX}"
    
    if [[ -f "$backup_file" ]]; then
        print_warning "Backup already exists: $(basename "$backup_file")"
        return 0
    fi
    
    if cp "$file" "$backup_file"; then
        print_success "Created backup: $(basename "$backup_file")"
    else
        print_error "Failed to create backup: $(basename "$backup_file")"
        return 1
    fi
}

# Function to check if file is already encrypted
is_encrypted() {
    local file="$1"
    if grep -q "pyarmor_runtime\|from pytransform\|__pyarmor__" "$file" 2>/dev/null; then
        return 0
    else
        return 1
    fi
}

# Function to encrypt a single file using PyArmor 9.x
encrypt_file() {
    local file="$1"
    local dir="$(dirname "$file")"
    local filename="$(basename "$file")"
    
    # Check if already encrypted
    if is_encrypted "$file"; then
        if [[ "$FORCE_MODE" == true ]]; then
            print_warning "File already encrypted but force mode enabled: $filename"
        else
            print_warning "File already encrypted: $filename"
            return 0
        fi
    fi
    
    print_info "Encrypting: $filename"
    
    # Create backup
    if ! create_backup "$file"; then
        return 1
    fi
    
    # Create temporary directory for encryption
    local temp_dir="${dir}/.pyarmor_temp_$$"
    if ! mkdir -p "$temp_dir"; then
        print_error "Failed to create temporary directory: $temp_dir"
        return 1
    fi
    
    # Initialize PyArmor project in temp directory
    print_debug "Initializing PyArmor project in temp directory..."
    if ! $PYARMOR_CMD init --src "$dir" --entry "$filename" "$temp_dir" &> /dev/null; then
        print_warning "PyArmor project initialization failed, trying direct obfuscation..."
    fi
    
    # Encrypt the file using PyArmor 9.x obfuscate command
    local encryption_success=false
    local encryption_output=""
    local encryption_error=""

    print_info "Attempting encryption with PyArmor 9.x..."

    # Try PyArmor 9.x gen command (correct for 9.x)
    print_debug "Trying: $PYARMOR_CMD gen -O \"$temp_dir\" \"$file\""
    if encryption_output=$($PYARMOR_CMD gen -O "$temp_dir" "$file" 2>&1); then
        encryption_success=true
        print_debug "PyArmor gen command succeeded"
    else
        encryption_error="$encryption_output"
        print_warning "PyArmor gen command failed, checking if partial success..."
        
        # Check if license error occurred but runtime files were created
        if echo "$encryption_output" | grep -q "out of license"; then
            print_error "PyArmor trial license has expired. Runtime files created but obfuscation failed."
            print_error "To fix this issue:"
            print_error "1. Purchase a PyArmor license, or"
            print_error "2. Use a different obfuscation tool, or"
            print_error "3. Contact PyArmor support for trial extension"
            return 1
        else
            print_error "PyArmor gen command failed with unknown error"
        fi
    fi

    if [[ "$encryption_success" == true ]]; then
        # Find the encrypted file in the output directory
        local encrypted_file=""
        if [[ -f "${temp_dir}/${filename}" ]]; then
            encrypted_file="${temp_dir}/${filename}"
        elif [[ -f "${temp_dir}/dist/${filename}" ]]; then
            encrypted_file="${temp_dir}/dist/${filename}"
        elif [[ -f "${temp_dir}/pyarmor_runtime_000000/${filename}" ]]; then
            encrypted_file="${temp_dir}/pyarmor_runtime_000000/${filename}"
        elif [[ -f "${dir}/${filename}" ]]; then
            # File was obfuscated in place
            encrypted_file="${dir}/${filename}"
        fi
        
        if [[ -n "$encrypted_file" ]] && [[ "$encrypted_file" != "$file" ]]; then
            # Move encrypted file back
            if mv "$encrypted_file" "$file"; then
                print_success "Encrypted: $filename"
            else
                print_error "Failed to move encrypted file: $filename"
                return 1
            fi
        elif [[ "$encrypted_file" == "$file" ]]; then
            print_success "Encrypted: $filename (in place)"
        else
            print_error "Encrypted file not found in expected locations"
            print_debug "Contents of temp directory:"
            ls -la "$temp_dir" || true
            return 1
        fi
        
        # Copy runtime files if they don't exist (PyArmor 9.x format)
        local runtime_found=false
        for runtime_candidate in "${RUNTIME_DIR}" "pytransform" "pyarmor_runtime" "dist"; do
            if [[ -d "${temp_dir}/${runtime_candidate}" ]] && [[ ! -d "${dir}/${runtime_candidate}" ]]; then
                if cp -r "${temp_dir}/${runtime_candidate}" "${dir}/"; then
                    print_success "Created runtime directory: ${runtime_candidate}/"
                    runtime_found=true
                    break
                else
                    print_warning "Failed to copy runtime directory: ${runtime_candidate}/"
                fi
            fi
        done

        if [[ "$runtime_found" == false ]]; then
            print_warning "No runtime directory found in temp directory"
            print_debug "Contents of temp directory:"
            ls -la "$temp_dir" | head -10 || true
        fi
    else
        print_error "PyArmor encryption failed for: $filename"
        if [[ -n "$encryption_error" ]]; then
            print_error "PyArmor error: $encryption_error"
        fi
        return 1
    fi
    
    # Clean up temporary directory
    rm -rf "$temp_dir"
}

# Function to restore from backup
restore_file() {
    local file="$1"
    local backup_file="${file}${BACKUP_SUFFIX}"
    
    if [[ -f "$backup_file" ]]; then
        if mv "$backup_file" "$file"; then
            print_success "Restored from backup: $(basename "$file")"
        else
            print_error "Failed to restore from backup: $(basename "$file")"
            return 1
        fi
    else
        print_error "No backup found for: $(basename "$file")"
        return 1
    fi
}

# Function to test if encrypted file works
test_encrypted_file() {
    local file="$1"
    local dir="$(dirname "$file")"
    local filename="$(basename "$file")"
    
    print_info "Testing encrypted file: $filename"
    
    # Change to the file's directory to ensure proper imports
    local original_dir=$(pwd)
    cd "$dir"
    
    # Test if the file can be imported without running the main code
    local test_result
    if test_result=$(python3 -c "
import sys
import os
sys.path.insert(0, '.')
try:
    # Try to compile the file first (syntax check)
    with open('$filename', 'r') as f:
        code = f.read()
    compile(code, '$filename', 'exec')
    
    # Try to import the module (imports check)
    if '$filename' == 'server.py':
        import importlib.util
        spec = importlib.util.spec_from_file_location('server', '$filename')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif '$filename' == 'lol.py':
        import importlib.util
        spec = importlib.util.spec_from_file_location('lol', '$filename')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    
    print('SUCCESS')
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
" 2>&1); then
        if echo "$test_result" | grep -q "SUCCESS"; then
            print_success "✅ $filename works correctly after encryption"
            cd "$original_dir"
            return 0
        else
            print_error "❌ $filename failed to work after encryption"
            print_error "Test output: $test_result"
            cd "$original_dir"
            return 1
        fi
    else
        print_error "❌ $filename failed to work after encryption"
        print_error "Test output: $test_result"
        cd "$original_dir"
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "PyArmor 9.x Specific Files Encryption Script v${SCRIPT_VERSION}"
    echo ""
    echo "Usage: $0 [OPTIONS] <directory>"
    echo ""
    echo "Encrypts server.py files in a directory using PyArmor 9.x"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help          Show this help message"
    echo "  -d, --dry-run       Show what would be encrypted without doing it"
    echo "  -r, --restore       Restore files from backup (undo encryption)"
    echo "  -l, --list          List files that would be encrypted"
    echo "  -f, --force         Force encryption even if files appear encrypted"
    echo "  -t, --test          Test if encrypted files work correctly"
    echo "  -v, --verbose       Enable verbose output"
    echo ""
    echo "EXAMPLES:"
    echo "  $0 /path/to/repo                    # Encrypt server.py"
    echo "  $0 -d /path/to/repo                 # Dry run - show what would be done"
    echo "  $0 -r /path/to/repo                 # Restore from backups"
    echo "  $0 -l /path/to/repo                 # List files that would be encrypted"
    echo "  $0 -t /path/to/repo                 # Test if encrypted files work"
    echo "  $0 -v /path/to/repo                 # Verbose output"
    echo ""
    echo "REQUIREMENTS:"
    echo "  - PyArmor 9.x installed (pip install pyarmor==9.1.8)"
    echo "  - Python 3.8+"
    echo "  - Write access to target directory"
    echo ""
    echo "NOTES:"
    echo "  - Backups are created with .backup extension"
    echo "  - Runtime files are automatically copied"
    echo "  - Script supports multiple PyArmor command syntaxes"
    echo ""
}

# Function to show script info
show_info() {
    echo "PyArmor 9.x Specific Files Encryption Script v${SCRIPT_VERSION}"
    echo "Target files: ${TARGET_FILES[*]}"
    echo "Runtime directory: $RUNTIME_DIR"
    echo "Backup suffix: $BACKUP_SUFFIX"
    echo ""
}

# Main function
main() {
    local target_dir=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            -r|--restore)
                RESTORE_MODE=true
                shift
                ;;
            -l|--list)
                LIST_MODE=true
                shift
                ;;
            -f|--force)
                FORCE_MODE=true
                shift
                ;;
            -t|--test)
                TEST_MODE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --version)
                show_info
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                if [[ -z "$target_dir" ]]; then
                    target_dir="$1"
                else
                    print_error "Multiple directories specified"
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # Check if directory was provided
    if [[ -z "$target_dir" ]]; then
        print_error "No directory specified"
        show_usage
        exit 1
    fi
    
    # Convert to absolute path
    target_dir="$(realpath "$target_dir")"
    
    print_info "Target directory: $target_dir"
    
    # Check directory
    check_directory "$target_dir"
    
    # Check files exist
    check_files "$target_dir"
    
    # List mode
    if [[ "$LIST_MODE" == true ]]; then
        print_info "Files that would be encrypted:"
        for file in "${TARGET_FILES[@]}"; do
            local full_path="${target_dir}/${file}"
            if [[ -f "$full_path" ]]; then
                if is_encrypted "$full_path"; then
                    echo "  - $file (already encrypted)"
                else
                    echo "  - $file"
                fi
            else
                echo "  - $file (not found)"
            fi
        done
        exit 0
    fi
    
    # Test mode
    if [[ "$TEST_MODE" == true ]]; then
        print_info "Testing encrypted files..."
        local test_failed=false
        for file in "${TARGET_FILES[@]}"; do
            local full_path="${target_dir}/${file}"
            if [[ -f "$full_path" ]]; then
                if ! test_encrypted_file "$full_path"; then
                    test_failed=true
                fi
            fi
        done
        
        if [[ "$test_failed" == true ]]; then
            print_error "Some encrypted files failed testing"
            exit 1
        else
            print_success "All encrypted files passed testing"
            exit 0
        fi
    fi
    
    # Restore mode
    if [[ "$RESTORE_MODE" == true ]]; then
        print_info "Restoring files from backups..."
        local restore_failed=false
        for file in "${TARGET_FILES[@]}"; do
            local full_path="${target_dir}/${file}"
            if [[ -f "$full_path" ]]; then
                if ! restore_file "$full_path"; then
                    restore_failed=true
                fi
            fi
        done
        
        if [[ "$restore_failed" == true ]]; then
            print_error "Some files failed to restore"
            exit 1
        else
            print_success "All files restored successfully"
            exit 0
        fi
    fi
    
    # Check PyArmor installation
    check_pyarmor
    
    # Dry run mode
    if [[ "$DRY_RUN" == true ]]; then
        print_info "DRY RUN MODE - No files will be modified"
        print_warning "This will encrypt ${#TARGET_FILES[@]} file(s) in: $target_dir"
        print_info "Files to encrypt: ${TARGET_FILES[*]}"
        print_warning "Backups will be created with $BACKUP_SUFFIX extension"
        print_info "Continue? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            print_info "Dry run completed - would proceed with encryption"
        else
            print_info "Dry run cancelled"
        fi
        exit 0
    fi
    
    # Confirm encryption
    print_warning "This will encrypt ${#TARGET_FILES[@]} file(s) in: $target_dir"
    print_info "Files to encrypt: ${TARGET_FILES[*]}"
    print_warning "Backups will be created with $BACKUP_SUFFIX extension"
    print_info "Continue? (y/N): "
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        print_info "Encryption cancelled"
        exit 0
    fi
    
    # Encrypt files
    print_info "Starting encryption process..."
    local encryption_failed=false
    for file in "${TARGET_FILES[@]}"; do
        local full_path="${target_dir}/${file}"
        if [[ -f "$full_path" ]]; then
            if ! encrypt_file "$full_path"; then
                encryption_failed=true
            fi
        fi
    done
    
    if [[ "$encryption_failed" == true ]]; then
        print_error "Some files failed to encrypt"
        exit 1
    else
        print_success "All files encrypted successfully"
        
        # Test encrypted files
        print_info "Testing encrypted files..."
        local test_failed=false
        for file in "${TARGET_FILES[@]}"; do
            local full_path="${target_dir}/${file}"
            if [[ -f "$full_path" ]]; then
                if ! test_encrypted_file "$full_path"; then
                    test_failed=true
                fi
            fi
        done
        
        if [[ "$test_failed" == true ]]; then
            print_warning "Some encrypted files failed testing - check the output above"
        else
            print_success "All encrypted files passed testing"
        fi
    fi
}

# Run main function with all arguments
main "$@"


