from pysqlite2 import dbapi2 as sqlite

con = sqlite.connect('test.db')
print(con.execute('select count (filename) from imlist').fetchone())
print(con.execute('select * from imlist').fetchone())