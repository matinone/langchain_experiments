import requests


def scrape_linkedin_profile(profile_url: str):
    """
    Manually scrape information from a LinkedIn profile.
    """

    # This should be done using Proxycurl each time:
    # (https://nubela.co/proxycurl/docs?python#people-api-person-profile-endpoint)

    # api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    # api_key = "YOUR_API_KEY"
    # header_dic = {"Authorization": "Bearer " + api_key}
    # response = requests.get(api_endpoint, params={"url": profile_url}, headers=header_dic)
    # return response

    # But to save credits, we will use the Proxycurl API only once, upload the response
    # to a Github gist, and query that information instead.
    gist_url = "https://gist.githubusercontent.com/emarco177/0d6a3f93dd06634d95e46a2782ed7490/raw/fad4d7a87e3e934ad52ba2a968bad9eb45128665/eden-marco.json"
    response_json = requests.get(gist_url).json()

    # remove empty fields
    data = {
        key: value
        for key, value in response_json.items()
        if value not in ([], "", None)
        and key not in ["people_also_viewed", "certifications"]
    }
    # remove profile_pic_url from groups
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data
