import pickle
from pysqlite2 import dbapi2 as sqlite

class Indexer(object):
    def __init__(self,db,voc):
        """ Initialize with the name of the database
        and a vocabulary object. """
        self.con = sqlite.connect(db)
        self.voc = voc
    def __del__(self):
        self.con.close()
    def db_commit(self):
        self.con.commit()

def create_tables(self):
    """ Create the database tables. """
    self.con.execute('create table imlist(filename)')
    self.con.execute('create table imwords(imid,wordid,vocname)')
    self.con.execute('create table imhistograms(imid,histogram,vocname)')
    self.con.execute('create index im_idx on imlist(filename)')
    self.con.execute('create index wordid_idx on imwords(wordid)')
    self.con.execute('create index imid_idx on imwords(imid)')
    self.con.execute('create index imidhist_idx on imhistograms(imid)')
    self.db_commit()


def add_to_index(self,imname,descr):
    """ Take an image with feature descriptors,
    project on vocabulary and add to database. """
    if self.is_indexed(imname): return
    print('indexing', imname)
    # get the imid
    imid = self.get_id(imname)
    # get the words
    imwords = self.voc.project(descr)
    nbr_words = imwords.shape[0]
    # link each word to image
    for i in range(nbr_words):
        word = imwords[i]
        # wordid is the word number itself
        self.con.execute("insert into imwords(imid,wordid,vocname)values (?,?,?)", (imid,word,self.voc.name))
    # store word histogram for image
    # use pickle to encode NumPy arrays as strings
    self.con.execute("insert into imhistograms(imid,histogram,vocname)values (?,?,?)", (imid,pickle.dumps(imwords),self.voc.name))

def is_indexed(self,imname):
    """ Returns True if imname has been indexed. """
    im = self.con.execute("select rowid from imlist wherefilename='%s'" % imname).fetchone()
    return im != None

def get_id(self,imname):
    """ Get an entry id and add if not present. """
    cur = self.con.execute("select rowid from imlist where filename='%s'" % imname)
    res=cur.fetchone()
    if res==None:
        cur = self.con.execute("insert into imlist(filename) values ('%s')" % imname)
        return cur.lastrowid
    else:
        return res[0]
    
class Searcher(object):
    def __init__(self,db,voc):
        """ Initialize with the name of the database. """
        self.con = sqlite.connect(db)
        self.voc = voc
    def __del__(self):
        self.con.close()