# Notes
## `neo4j`
### Auth
Auth is set in the system database, on db creation e.g. in the docker container
The `ENV` variable `NEO4J_AUTH` will configure the password setting on a newly-created db, e.g.

    NEO4J_AUTH=neo4j/some_new_passwd
    
If the database already exists, e.g. by being bind-mounted in docker, the user will already have
a password, which won't be changed by setting `NEO4J_AUTH` for startup. Once the password is set in the database,
if you want to change the passwd, tell the database:

    ALTER CURRENT USER SET PASSWORD FROM 'oldpass' TO 'newpass'
    
