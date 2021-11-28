import uvicorn


def main():
    uvicorn.run("FastAPI:app", port=8080, host='0.0.0.0', debug=True, reload=True)


if __name__ == "__main__":
    main()
