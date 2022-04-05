FROM python:3.8.6 as base

RUN apt-get update \
    && apt-get install -y vim less apt-utils curl apt-transport-https gnupg2 \
        git openssh-server \
    && curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/10/prod.list > \
        /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y unixodbc-dev msodbcsql17 mssql-tools

# Set time zone
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    dpkg-reconfigure --frontend noninteractive tzdata

# Create non-root user

# Enable bash syntax colouring.
ENV TERM xterm-256color

# poetry path to install
ENV POETRY_HOME="/opt/poetry"
ENV PATH="${PATH}:$POETRY_HOME/bin"

# Install poetry
WORKDIR /home/root
ENV PYTHONPATH=/home/root
RUN curl -sSL https://install.python-poetry.org | python3 -
# RUN pip install poetry
ENV PATH = "${PATH}:/root/.local/bin"
RUN echo 'export PS1="🐳 \[\033[1;36m\][\u@docker] \[\033[1;34m\]\W\[\033[0;35m\] # \[\033[0m\]"' >> ~/.bashrc


# Dalib
COPY . ./app
WORKDIR /home/root/app
RUN poetry install #--no-dev

# switch to root to circumvent permission error
EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "order_generator.api.api:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000", "--reload"]
