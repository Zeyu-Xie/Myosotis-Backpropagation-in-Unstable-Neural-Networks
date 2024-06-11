FROM acanxie/xelatex:1.1.0-linux-arm64

WORKDIR /app

COPY . /app

RUN chmod +x /app/compile.sh

CMD ["bash", "/app/compile.sh"]