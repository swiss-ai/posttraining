## Storage permissions to share directories

To remove.

To share with specific users

```bash
# For shared directories with other members
cd dirname
chmod -R g+rw .
chmod -R g+s .
for usr in smoalla smoalla; do
    setfacl -R -L -m u:$usr:rwx .
    setfacl -R -L -d -m u:$usr:rwx .
done
```
