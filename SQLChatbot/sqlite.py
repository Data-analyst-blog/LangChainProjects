#### Intention of this notebook is to create a sqlite3 database without sql server support
import sqlite3

## Code to connect to SQL Database
connection = sqlite3.connect('students.db')

## Create a cursor object to insert record , create table 
cursor=connection.cursor()

## Create the table
table_info="""
create table STUDENT(NAME VARCHAR(25), CLASS VARCHAR(25),
SECTION VARCHAR(25), MARKS INT)"""

cursor.execute(table_info)      # To execute the table

## Insert some more records
cursor.execute("""Insert into STUDENT values('Yash','Data Science','A',90)""")
cursor.execute("""Insert into STUDENT values('Madhu','Data Science','B',100)""")
cursor.execute("""Insert into STUDENT values('Mukesh','Data Science','A',86)""")
cursor.execute("""Insert into STUDENT values('Jacob','DEVOPS','A',50)""")
cursor.execute("""Insert into STUDENT values('Dipesh','DEVOPS','A',35)""")

## Display all records
print("The inserted records are")
data = cursor.execute("""SELECT * FROM STUDENT""")
for row in data:
    print(row)

## Commit your changes in databases
connection.commit()
connection.close()