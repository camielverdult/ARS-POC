# CNN app

Use `./bin/install_mac` to install the pip requirements. To seed the database with domains to screenshot, you can run `python3 db.py seed`. You can now run `python3 get_screenshots.py`. This will go through all the domains in the database, make a screenshot and filter out CDN domains for future runs. To purge the database and screenshots, you can use `python3 db.py purge`. Topics can be set in the `topics.yml` file, these must be set before seeding the database.

# Next steps

- Labelling domains (fill labels table)
- Update train.py to use the db
- Comparison to YOLO model