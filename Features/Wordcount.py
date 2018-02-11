def runFeature(responseText):
    words = responseText.split(' ')
    return len([x for x in words if (x.strip() != '' and len(x.strip()) > 1)])