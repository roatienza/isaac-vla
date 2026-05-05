import tensorflow as tf
import sys
tf.config.set_visible_devices([], 'GPU')

tfrecord_path = 'datasets/rlds/libero_spatial_no_noops/1.0.0/train.tfrecord'
dataset = tf.data.TFRecordDataset(tfrecord_path)

for record in dataset.take(1):
    raw = record.numpy()
    parsed = tf.train.Example.FromString(raw)
    
    for k, v in parsed.features.feature.items():
        sys.stderr.write(f'Key: {k}\n')
        if v.HasField('bytes_list'):
            sys.stderr.write(f'  Type: bytes_list, values: {len(v.bytes_list.value)}\n')
            for i, val in enumerate(v.bytes_list.value[:2]):
                sys.stderr.write(f'    [{i}] len={len(val)}\n')
        elif v.HasField('float_list'):
            sys.stderr.write(f'  Type: float_list, values: {len(v.float_list.value)}\n')
        elif v.HasField('int64_list'):
            sys.stderr.write(f'  Type: int64_list, values: {len(v.int64_list.value)}\n')
        sys.stderr.write('\n')
    sys.stderr.flush()
    break
