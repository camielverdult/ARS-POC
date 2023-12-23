import get_screenshots
import train

def main():
    get_screenshots.screenshot_domains()
    train.train()

if __name__ == "__main__":
    main()