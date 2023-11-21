from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from flask_login import login_user, login_required, logout_user, current_user

auth = Blueprint("auth", __name__)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password1')

        user = User.query.filter_by(email=email).first()
        if user:  # user exists
            if check_password_hash(user.password, password):
                flash("logged in successfully", category='success')
                login_user(user, remember=True)  # 浏览器会记住user，不用每次都登录
                return redirect(url_for('views.home'))
            else:
                flash("incorrect password", category='error')
        else:
            flash("email does not exit", category='error')
    return render_template("login.html", user=current_user)


@auth.route('/logout')
@login_required  # 只有在登陆的状态下才额能访问logout界面
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        firstName = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')
        # check form valid or not
        user = User.query.filter_by(email=email).first()
        if user:
            flash('email already exists', category='error')
        elif len(email) < 4:
            flash('Email must be greater than 3 characters.', category='error')  # category的名字是用来区别popup信息颜色的
        elif len(firstName) < 2:
            flash('first name must be greater than 1 characters.', category='error')  # category的名字是用来区别popup信息颜色的
        elif password2 != password1:
            flash('passwords don\'t match', category='error')  # category的名字是用来区别popup信息颜色的
        elif len(password1) < 7:
            flash('passwords must be greater than 6 characters.', category='error')  # category的名字是用来区别popup信息颜色的
        else:
            new_user = User(email=email, first_name=firstName,
                            password=generate_password_hash(password1, method='sha256'))
            db.session.add(new_user)
            db.session.commit()
            login_user(user, remember=True)  # 注册完后自动登录
            flash('account created', category='success')  # category的名字是用来区别popup信息颜色的
            return redirect(url_for('views.home'))  # views是blueprint的名字，home是里面的function的名字

    return render_template("sign_up.html",user=current_user)
