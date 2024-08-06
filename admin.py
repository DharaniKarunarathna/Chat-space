
from werkzeug.security import generate_password_hash
from app import app, db, User

def create_admin_user():
    with app.app_context():
        # Create an admin user with hashed password
        admin = User(id='admin', password_hash=generate_password_hash('adminpassword'))
        db.session.add(admin)
        db.session.commit()
        print("Admin user created.")

if __name__ == '__main__':
    create_admin_user()
