import hashlib

def write_hash_to_file(filename, hash_value):
    with open(filename, 'wb') as f:
        f.write(hash_value)
        f.close()

def hash_file_calc(file_path):
    mach = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            mach.update(chunk)
    return mach.hexdigest()


def calculate_and_write_hash_to_file(file_path, out_file_path):
    hash_digest = hash_file_calc(file_path)
    write_hash_to_file(out_file_path, hash_digest)