import sys
import moxing as mox

src_data = sys.argv[1]
dst_data = sys.argv[2]
file_or_not = sys.argv[3]
print('src is %s' % src_data)
print('dst is %s' % dst_data)
print(file_or_not)
if file_or_not == 'True':
    print('exec the mox.file.copy')
    mox.file.copy(src_data, dst_data)
else:
    print('exec the mox.file.copy_parallel')
    mox.file.copy_parallel(src_data, dst_data)
print('finish copy')
