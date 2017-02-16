from flask import Flask, render_template, request, redirect, flash, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime as dt
from pybtex.database import parse_bytes

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/bibtex.db'
app.config['ALLOWED_EXTENSIONS'] = ['bib']
db = SQLAlchemy(app)

app.debug = True
app.secret_key = 'some_secret'


class Collection(db.Model):
    """Collection of bibliography items"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), unique=True)
    create_date = db.Column(db.String(30), unique=True)
    items = db.relationship('Item', backref='collection',
                            lazy='dynamic')

    def __init__(self, name):
        self.name = name
        self.create_date = str(dt.now())

    def __repr__(self):
        return 'Collection {} created at {}'.format(self.name, self.create_date)


class Item(db.Model):
    """Individual bibliography item"""
    id = db.Column(db.Integer, primary_key=True)
    citation_tag = db.Column(db.String())
    author_list = db.Column(db.String())
    editor_list = db.Column(db.String())
    journal_book = db.Column(db.String())
    volume = db.Column(db.String())
    pages = db.Column(db.String())
    year = db.Column(db.String())
    title = db.Column(db.String())
    collection_id = db.Column(db.Integer, db.ForeignKey('collection.id'))

    def __init__(self, citation_tag, author_list, editor_list, journal_book, volume, pages, year, title, collection):
        self.citation_tag = citation_tag
        self.author_list = author_list
        self.editor_list = editor_list
        self.journal_book = journal_book
        self.volume = volume
        self.pages = pages
        self.year = year
        self.title = title
        self.collection = collection

    def __repr__(self):
        return 'Tag: {}'.format(self.citation_tag)

db.create_all()
try:
    db.session.commit()
except:
    pass


@app.route("/")
def index():
    """Landing page shows all available collections and has links to uploading and querying page"""
    collections = [n[0] for n in db.session.query(Collection.name).all()]
    return render_template('index.html', collections=collections)


def allowed(filename):
    """Check if the uploaded file is a BibTeX file"""
    return '.' in filename and filename.split('.')[1] in app.config['ALLOWED_EXTENSIONS']


def journal_abbrev(journal):
    """Common BibTeX journal abbreviations"""
    abbrev = {'\\aj': 'Astronomical Journal',
              '\\aap': 'Astronomy and Astrophysics',
              '\\apj': 'Astrophysical Journal',
              '\mnras': 'Monthly Notices of the RAS',
              '\\araa': 'Annual Review of Astronomy and Astrophysics',
              '\\aaps': 'Astronomy and Astrophysics, Supplement'}
    return abbrev[journal]


def load(collname, bib_content):
    """Load uploaded data into the database"""
    coll = Collection(collname)
    db.session.add(coll)
    bib_data = parse_bytes(bib_content, bib_format="bibtex")
    for k, v in bib_data.entries.items():
        citation_tag = k
        title = v.fields['Title'].replace('{', '').replace('}', '')
        year = v.fields['Year']
        try:
            # If nothing in the 'Journal' tag, try 'BookTitle'
            journal_book = v.fields['Journal']
            try:
                journal_book = journal_abbrev(journal_book)
            except KeyError:
                pass
        except KeyError:
            try:
                journal_book = v.fields['BookTitle']
            except KeyError:
                journal_book = ''
        try:
            volume = v.fields['Volume']
        except KeyError:
            volume = ''
        try:
            pages = v.fields['Pages'].replace('+', '')
        except:
            pages = ''
        author_list, editor_list = get_author_list(v.persons)
        item = Item(citation_tag=citation_tag,
                    author_list=author_list,
                    editor_list=editor_list,
                    journal_book=journal_book,
                    volume=volume,
                    pages=pages,
                    year=year,
                    title=title,
                    collection=coll)
        db.session.add(item)
    db.session.commit()


def get_author_list(persons):
    """Turn a persons dictionary from pybtex into plain-text lists of authors and editors"""
    try:
        authors = persons['Author']
        authors = ', '.join([p.get_part_as_text('bibtex_first')+p.get_part_as_text('last') for p in authors])
        authors = authors.replace('{', ' ').replace('}', '')
        authors = authors.replace('~', ' ')
        authors = authors.replace('\\', '')
    except:
        authors = ''
    try:
        editors = persons['Editor']
        editors = ', '.join([p.get_part_as_text('bibtex_first')+p.get_part_as_text('last') for p in editors])
        editors = editors.replace('{', ' ').replace('}', '')
        editors = editors.replace('~', ' ')
        editors = editors.replace('\\', '')
    except:
        editors = ''
    return authors, editors


@app.route("/add", methods=['GET', 'POST'])
def add_collection():
    """Upload a new collection"""
    if request.method == 'POST':
        if request.form['collname'] in (None, ''):
            flash("Please provide a collection name.")
            return redirect(request.url)
        if 'file' not in request.files:
            flash("No file found.")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("Please select a BibTeX file to upload.")
            return redirect(request.url)
        if file:
            if allowed(file.filename):
                content = file.read()
                load(request.form['collname'], content)
                return redirect(url_for('index'))
            else:
                flash("Unsupported file type (.bib only)")
                return redirect(request.url)
    return render_template('insert.html')


@app.route("/query", methods=['GET', 'POST'])
@app.route("/query/<collection>", methods=['GET', 'POST'])
def query(collection='all'):
    """Query a specified collection. If no collection specified, query all"""
    results = []
    if request.method == 'POST':
        q = "select * from item inner join collection on item.collection_id = collection.id where "
        whereclause = request.form['query']
        if collection != 'all':
            collection_id = Collection.query.filter_by(name=collection).first().id
            q += "collection_id={} and ".format(collection_id)
        q += whereclause
        print(q)
        try:
            results = list(db.engine.execute(q))
        except:
            flash("Invalid SQL query. Try again.")
            return redirect(request.url)
    return render_template('query.html', collection=collection, results=results)

if __name__ == "__main__":
    app.run()
