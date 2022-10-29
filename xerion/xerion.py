# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from six.moves.urllib import request

import glob
import os
import platform  # Mac or Linux special for uncompress command
import errno
import sys
import numpy as np
import codecs
import re
import subprocess
import sys
import tarfile
import matplotlib.pyplot as plt

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

######################################################################
# Search Path
######################################################################

path = []
"""A list of directories where the xerion data package might exist.
   These directories will be checked in order when looking for a
   resource in the data package.  Note that this allows users to
   substitute in their own versions of resources, if they have them
   (e.g., in their home directory under ~/xerion_data)."""

# User-specified locations:
_paths_from_env = os.environ.get('XERION_DATA', str('')).split(os.pathsep)
path += [d for d in _paths_from_env if d]

if sys.platform.startswith('win'):
    # Common locations on Windows:
    path += [
        os.path.join(sys.prefix, str('xerion_data')),
        os.path.join(sys.prefix, str('share'), str('xerion_data')),
        os.path.join(sys.prefix, str('lib'), str('xerion_data')),
        os.path.join(os.environ.get(str('APPDATA'), str('C:\\')), str('xerion_data')),
        str(r'C:\xerion_data'),
        str(r'D:\xerion_data'),
        str(r'E:\xerion_data'),
    ]
else:
    # Common locations on UNIX & OS X:
    path += [
        os.path.join(sys.prefix, str('xerion_data')),
        os.path.join(sys.prefix, str('share'), str('xerion_data')),
        os.path.join(sys.prefix, str('lib'), str('xerion_data')),
        str('/usr/share/xerionk_data'),
        str('/usr/local/share/xerion_data'),
        str('/usr/lib/xerion_data'),
        str('/usr/local/lib/xerion_data'),
    ]



class Xerion(object):
    """Managing xerion datafiles.

    Read datafiles ending with '-nsyl.ex' and '-syl.ex'
    from `xerion_prefix/datadir`, and 
    save them to `pkl_dir` as pickle files.

    Usage:
    ```python
    print(Xerion().Orthography)  # for input data format
    print(Xerion().Phonology)    # for output data format
    X = xerion().input
    y = xerion().output
    ```

    The original datafiles can be obtained from http://www.cnbc.cmu.edu/~plaut/xerion/
    """

    def __init__(self,
                 download_dir=None,
                 basefilename='SM-nsyl',
                 xerion_prefix = 'nets/share/',
                 datadir=None,
                 pkl_dir=None,
                 remake=False, readall=False, saveall=False,
                 forceDownload=False):
        self.module_path = os.path.dirname(__file__)
        self.xerion_prefix = xerion_prefix
        self.origina_xerion_prefix = 'nets/share/'
        self.basefilename = basefilename

        if datadir == None:
            self.datadir = self.module_path + '/data/'
            pkl_dir = self.datadir
        else:
            self.datadir = datadir
        if self.datadir[-1] != '/':
            self.datadir += '/'
        #print('self.datadir={}'.format(self.datadir))
        if pkl_dir == None:
            pkl_dir = self.datadir
        self.pkl_dir = pkl_dir
        if self.pkl_dir[-1] != '/':
            self.pkl_dir += '/'
        self.pkl_file = pkl_dir + self.basefilename + '.pkl'
        self.datafilename = self.pkl_file
        #print('self.datafilename={}'.format(self.datafilename))
        self.tags = ('#', 'seq', 'grapheme', 'phoneme', 'freq',
                     'tag', 'inputs', 'outputs')
        self.dbs = {}
        self.db = self.load_pickle(filename=self.datafilename)
        self.inputs = self.db['inputs']
        self.outputs = self.db['outputs']
        self.seq = self.db['seq']
        self.freq = self.db['freq']
        self.grapheme = self.db['grapheme']
        self.phoneme = self.db['phoneme']
        self.tag = self.db['tag']
        self.dbs[self.basefilename] = self.db

        if remake:
            self.dbs = self.make_all()
            saveall = True
        if saveall == True:
            self.save_all()
            readall = True
        if readall:
            self.dbs = self.read_all()

        self.url_base = 'http://www.cnbc.cmu.edu/~plaut/xerion/'
        self.url_file = 'xerion-3.1-nets-share.tar.gz'
        self.origfile_size = 1026691
        self.syl_files = ['SM-syl.ex', 'besnerNW-syl.ex',
                          'bodies-syl.ex', 'bodiesNW-syl.ex',
                          'friedmanNW-syl.ex', 'glushkoNW-syl.ex', 'graphemes-syl.ex',
                          'jared1-syl.ex', 'jared2-syl.ex', 'megaNW-syl.ex',
                          'pureNW-syl.ex', 'surface-syl.ex', 'taraban-syl.ex',
                          'tarabanALL-syl.ex', 'tarabanEvN-syl.ex', 'tarabanNRE-syl.ex',
                          'vcoltheartNW-syl.ex']
        self.nsyl_files = ['SM-nsyl.ex', 'besnerNW-nsyl.ex', 'glushkoNW-nsyl.ex',
                          'graphemes-nsyl.ex', 'jared1-nsyl.ex', 'jared2-nsyl.ex',
                          'markPH-nsyl.ex', 'megaNW-nsyl.ex', 'surface-nsyl.ex',
                          'taraban-nsyl.ex', 'tarabanALL-nsyl.ex', 'tarabanEvN-nsyl.ex',
                          'tarabanNRE-nsyl.ex']
        self.datafilenames = [ *self.nsyl_files, *self.syl_files]
        self.Orthography={'onset':['Y', 'S', 'P', 'T', 'K', 'Q', 'C', 'B', 'D', 'G',
                                   'F', 'V', 'J', 'Z', 'L', 'M', 'N', 'R', 'W', 'H',
                                   'CH', 'GH', 'GN', 'PH', 'PS', 'RH', 'SH', 'TH',
                                   'TS', 'WH'],
                          'vowel':['E', 'I', 'O', 'U', 'A', 'Y', 'AI', 'AU', 'AW',
                                   'AY', 'EA', 'EE', 'EI', 'EU', 'EW', 'EY', 'IE',
                                   'OA', 'OE', 'OI', 'OO', 'OU', 'OW', 'OY', 'UE',
                                   'UI', 'UY'],
                          'coda':['H', 'R', 'L', 'M', 'N', 'B', 'D', 'G', 'C', 'X',
                                  'F', 'V', '∫', 'S', 'Z', 'P', 'T', 'K', 'Q', 'BB',
                                  'CH', 'CK', 'DD', 'DG', 'FF', 'GG', 'GH', 'GN',
                                  'KS', 'LL',
                                  'NG', 'NN', 'PH', 'PP', 'PS', 'RR', 'SH', 'SL',
                                  'SS', 'TCH',
                                  'TH', 'TS', 'TT', 'ZZ', 'U', 'E', 'ES', 'ED']}
        self.Phonology={'onset':['s', 'S', 'C', 'z', 'Z', 'j', 'f', 'v', 'T', 'D',
                                 'p', 'b', 't', 'd', 'k', 'g', 'm', 'n', 'h', 'I',
                                 'r', 'w', 'y'],
                        'vowel': ['a', 'e', 'i', 'o', 'u', '@', '^', 'A', 'E', 'I',
                                  'O', 'U', 'W', 'Y'],
                        'coda':['r', 'I', 'm', 'n', 'N', 'b', 'g', 'd', 'ps', 'ks',
                                'ts', 's', 'z', 'f', 'v', 'p', 'k', 't', 'S', 'Z',
                                'T', 'D', 'C', 'j']}


    #def read_a_xerion_file(filename='SM-nsyl.pkl'):
    #    pass

    def read_all(self):
        """reading data files named ening with '-nsyl.ex'."""
        dbs = {}
        for dname in self.datafilenames:
            dname_ = re.sub('.ex', '', dname)
            filename = self.pkl_dir + dname_ + '.pkl'
            if not os.path.isfile(filename):
                raise ValueError('{0} could not found'.format(filename))
            dbs[dname_] = self.load_pickle(filename=self.pkl_file)
        return dbs

    def save_db(self, filename, db):
        """save a xerion pickle file."""
        dirname = self.pkl_dir
        if not os.path.exists(self.pkl_dir):
            os.makedirs(self.pkl_dir)
            if not os.path.exists(self.pkl_dir):
                raise OSError('{} was not found'.format(self.pkl_dir))
        dest_filename = self.pkl_dir + re.sub('.ex', '.pkl', filename)
        print('dest_filename={}'.format(dest_filename))
        with codecs.open(dest_filename, 'wb') as f:
            pickle.dump(db, f)


    def save_all(self):
        """saving data files to be pickled."""
        dirname = self.pkl_dir
        if not os.path.exists(self.pkl_dir):
            os.makedirs(self.pkl_dir)
            if not os.path.exists(self.pkl_dir):
                raise OSError('{} was not found'.format(self.pkl_dir))
        for db in self.dbs:
            dest_filename = self.pkl_dir + re.sub('.ex', '.pkl', db)
            with codecs.open(dest_filename, 'wb') as f:
                pickle.dump(self.dbs[db], f)


    def load_pickle(self, filename=None):
        if filename == None:
            filename=self.datafilename
        if not os.path.isfile(filename):
            filename = self.pkl_dir + filename
        if not os.path.isfile(filename):
            raise ValueError('Could not find {}'.format(filename))
        with open(filename, 'rb') as f:
            db = pickle.load(f)
        return db


    def make_all(self):
        dbs = {}
        for dname in self.datafilenames:
            #filename = self.datadir + self.xerion_prefix + dname
            #if not os.path.isfile(filename):
            #    print('{0} could not found'.format(filename))
            #    downfilename, h = self.download()
            #    self.extractall()
            #dbs[dname.split('.')[0]] = self.read_xerion(filename)
            dbs[dname.split('.')[0]] = self.read_xerion(dname)
            #self.save_db(dname)
        self.dbs = dbs
        return self.dbs


    #def read_xerion(self, filename="SM-nsyl.ex"):
    def read_xerion(self, filename):
        if not os.path.isfile(filename):
            filename = self.datadir + filename
        if not os.path.isfile(filename):
            raise ValueError('Could not find {}'.format(filename))
        with codecs.open(filename,'r') as f:
            lines = f.readlines()

        inp_flag = False
        inputs, outputs, grapheme, phoneme = {}, {}, {}, {}
        freq, tags, seqs = {}, {}, {}
        for i, line in enumerate(lines[1:]):
            if len(line) == 0:
                continue
            a = line.strip().split(' ')
            if line[0] == '#':
                if a[0] == '#WARNING:':
                    continue
                try:
                    seq = int(a[self.tags.index('seq')])
                except:
                    continue
                _grapheme = a[self.tags.index('grapheme')]
                _phoneme = a[self.tags.index('phoneme')]
                _freq = a[self.tags.index('freq')]
                _tag = a[self.tags.index('tag')]
                inp_flag = True
                if not seq in inputs:
                    inputs[seq] = list()
                    outputs[seq] = list()
                    grapheme[seq] = _grapheme
                    phoneme[seq] = _phoneme
                    freq[seq] = _freq
                    tags[seq] = _tag
                    seqs[seq] = seq
                continue
            elif line[0] == ',':
                inp_flag = False
                continue
            elif line[0] == ';':
                inp_flag = True
                continue
            if inp_flag:
                #print('hoge seq=', seq)
                for x in a:
                    try:
                        inputs[seq].append(int(x))
                    except:
                        pass  #print(x, end=', ')
            else:
                for x in a:
                    try:
                        outputs[seq].append(int(x))
                    except:
                        pass
            continue

        _inputs = np.array([inputs[seq] for seq in inputs], dtype=np.int16)
        _outputs = np.array([outputs[seq] for seq in outputs], dtype=np.int16)
        _grapheme = np.array([grapheme[seq] for seq in grapheme], dtype=np.unicode_)
        _phoneme = np.array([phoneme[seq] for seq in phoneme], dtype=np.unicode_)
        _freq = np.array([freq[seq] for seq in freq], dtype=np.float32)
        _tag = [tags[seq] for seq in tags]
        _seq = np.array([seqs[seq] for seq in seqs], dtype=np.int16)
        db = {'inputs': _inputs,
              'outputs': _outputs,
              'grapheme': _grapheme,
              'phoneme': _phoneme,
              'freq': _freq,
              'tag': _tag,
              'seq': _seq
        }
        return db
        #return _inputs, _outputs, _grapheme, _phoneme, _freq, _seqs

    def download(self, forcedownload=False, destdir=None):
        if destdir is None:
            destdir = self.datadir 
        if not os.path.exists(destdir):
            os.mkdir(destdir)
        dest_filename = destdir + self.url_file
        if os.path.exists(dest_filename):
            statinfo = os.stat(dest_filename)
            if statinfo.st_size != self.origfile_size:
                forceDownload = True
                print("File {} not expected size, forcing download".format(dest_filename))
            else:
                print("File '{}' allready downloaded.".format(dest_filename))
        if forcedownload == True or not os.path.exists(dest_filename):
            print('Attempting to download: {}'.format(dest_filename)) 
            print('From {}'.format(self.url_base + self.url_file))
            fname, h = request.urlretrieve(self.url_base+self.url_file, dest_filename)
            print("Downloaded '{}' successfully".format(dest_filename))
            return fname, h
        else:
            return dest_filename, None


    @staticmethod
    def extractall(gzfile=None):
        if gzfile is None:
            gzfile, _ = self.download()
        with tarfile.open(name=gzfile, mode='r:gz') as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=self.datadir)
        if platform.system() == 'Darwin':
            cmd = '/usr/bin/uncompress'
            args = self.datadir + self.xerion_prefix + '*.ex.Z'
            files = glob.glob(args)
            for file in sorted(files):
                print(cmd, file)
                try:
                    subprocess.Popen([cmd, file])
                except:
                    print('cmd {0} {1} failed'.format(cmd, file))
                    sys.exit()
            print('#extractall() completed. command:{}'.format(cmd))
        else:
            print('You must on Linux or Windows, Please uncompress manually')
            sys.exit()
        self.pkl_dir = self.datadir + self.xerion_prefix


    def note(self):
        print('\n\n# xerion() is the data management tool for PMSP96')
        print('# The original data will be found at:',
              self.url_base + self.url_file)
        print('# The data format is as following:')
        for l in [self.Orthography, self.Phonology]:
            for x in l:
                print(x, l[x])


    @staticmethod
    def usage():
        print('```python')
        print('import numpy')
        print('from xerion import Xerion')
        print()
        print('from sklearn.neural_network import MLPRegressor')
        print()
        print('dataset = Xerion()')
        print('X = dataset.inputs')
        print('y = data.outputs')
        print()
        print('model = MLPRegressor()')
        print('model.fit(X, y)')
        print('model.score(X, y)')
        print('```')

        
    def descr(self):
        fdescr_name = os.path.join(self.module_path, 'descr', 'xerion.md')
        print('self.module_path={}'.format(self.module_path))
        print('fdescr_name={}'.format(fdescr_name))

        with codecs.open(fdescr_name, 'r') as markdownfile:
            fdescr = markdownfile.read()
        print(fdescr)
