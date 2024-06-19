FROM python:3.12.3

RUN pip install modin[all]

COPY run_modin_in_process.py .

CMD ["python", "run_modin_in_process.py"]
