Schema
API Schema.

#Location
The base URL for all endpoints is https://api.coverr.co. You will use it in combination with all endpoints' paths.

For example, the videos endpoint will have this url: https://api.coverr.co/videos.

#Responses
The API returns information in JSON format.

#Pagination
Most endpoints split long response data into multiple pages.

You can control it with the page (zero based) and page_size query parameters.

#Paginated response example:
{
  "page": 0,       // current page
  "pages": 7,      // number of pages
  "page_size": 20, // number of results per page
  "total": 123,    // number of all results
  "hits": [        // current page data slice
    ...
  ]
}

Authentication
All of our API endpoints require authentication. As mentioned above — to access Coverr’s Free Videos API, you’ll need an API Key.

This key can be passed as an api_key url query parameter or Authorization http header.

#Making our first call
No matter the authentication method you choose (parameter or http header), once you get an API Key, you’re good to go!

Using the key you’ve received and the auth method you’ve chosen, you can now access our API:

#Method 1: Auth with query parameter
Try to hit this URL with your key:

https://api.coverr.co/videos?api_key={api_key}
#Method 2: Auth with HTTP Header
Use Postman or curl to try this request using your API key (notice the Bearer prefix):

curl -v \
  -H 'Authorization: Bearer {api_key}' \
  https://api.coverr.co/videos
Easy, right?

Next, we’ll start reviewing the main endpoints you can query.

* Again, to get an API key, simply contact us at team@coverr.co and describe your use-case in a few sentences.

Listing free videos
Listing videos is usually the beginning of every user journey in our system, and will most likely be the most useful query. We allow you to list top videos (by date or popular), search videos, or show all videos in a category or a collection.

#List videos
In order to list a general list of top videos, all you need to do is query this endpoint:

GET /videos
#Parameters
Param	Description
page	Page number. Number. Default: 0
page_size	Number of videos per page. Number. Default: 20
query	Search videos by query. String. Default: '', see a note below
sort	How to sort videos. String. Valid values: date, popular. Default: popular
urls	Add urls property in the response. Boolean. Default: false
#Response
{
  "page": 0,
  "pages": 50,
  "page_size": 20,
  "total": 3602,
  "hits": [
    {
      "id": "S1YbPl1NfI",
      "created_at": "2020-11-12T08:55:42.710Z",
      "updated_at": "2020-11-12T08:55:42.710Z",
      "title": "Cutting Wood Building Material With a Circular Electric Saw",
      "poster": "https://storage.coverr.co/p/QatsCWWAorI71sZ33DkHZREGWruZCHsg",
      "thumbnail": "https://storage.coverr.co/t/QatsCWWAorI71sZ33DkHZREGWruZCHsg",
      "description": "A close-up of the circular miter saw lowers and cuts off material",
      "is_vertical": false,
      "tags": [
        "circular saw",
        "miter saw",
        "saw"
      ],
      "downloads": 4,
      "views": 1,
      "published_at": "2020-11-15T09:27:18.578Z",
      "aspect_ratio": "16:9",
      "duration": 11.625,
      "max_height": 1152,
      "max_width": 2048,
      "urls": {
        "mp4": "https://storage.coverr.co/videos/QatsCWWAorI71sZ33DkHZREGWruZCHsg?token={token}",
        "mp4_preview": "https://storage.coverr.co/videos/QatsCWWAorI71sZ33DkHZREGWruZCHsg/preview?token={token}",
        "mp4_download": "https://storage.coverr.co/videos/QatsCWWAorI71sZ33DkHZREGWruZCHsg/download?token={token}&filename=Cutting Wood Building Material With a Circular Electric Saw"
      }
    }
  ]
}
#Search videos
Although getting a random list of videos is fun, it's kind of useless in most instances. Search is undoubtedly the most common API query and you'll probably use it extensively throughout the integration.

Search is simply done by adding the query parameter to videos endpoint:

GET /videos?query={search term goes here}
Nice, so we have a list of videos, but how can we get a single video’s details? We like how you think! Read on for all the answers to your burning questions.

#Get a video
Get a single video by id:

GET /videos/:id
#Parameters
Param	Description
id	Video id
#Response
{
  "id": "S1YbPl1NfI",
  "created_at": "2020-11-12T08:55:42.710Z",
  "updated_at": "2020-11-12T08:55:42.710Z",
  "title": "Cutting Wood Building Material With a Circular Electric Saw",
  "poster": "https://storage.coverr.co/p/QatsCWWAorI71sZ33DkHZREGWruZCHsg",
  "thumbnail": "https://storage.coverr.co/t/QatsCWWAorI71sZ33DkHZREGWruZCHsg",
  "description": "A close-up of the circular miter saw lowers and cuts off material",
  "is_vertical": false,
  "tags": [
    "circular saw",
    "miter saw",
    "saw"
  ],
  "downloads": 4,
  "views": 1,
  "published_at": "2020-11-15T09:27:18.578Z",
  "aspect_ratio": "16:9",
  "duration": 11.625,
  "max_height": 1152,
  "max_width": 2048,
  "urls": {
    "mp4": "",
    "mp4_preview": "",
    "mp4_download": ""
  }
}
#Download a video
To download a video you can use the url from the mp4_download field on the urls object. This will return the same mp4 file but with an additional Content-Disposition http response header that will indicate that the content is expected to be downloaded and saved locally and the browser won't try to inline it. In this case no additional action is required.

However, in most use cases you'd want your app users to view the video prior to downloading it, right? Therefore, at the point of the download event, you should already have the video file on hand. Or you just "select" a video in your app to be hotlinked later, for example as your website's main video.

In such cases, all we ask is for you to ping the download url so we can register in our stats that this video was indeed downloaded and not just viewed.

PATCH /videos/:id/stats/downloads
#Parameters
Param	Description
id	Video id
#Response
204 No Content
Notice:pinging the mp4_download is a MUST and not optional since it really improves our feedback loop and guides future content creation and video popularity rankings. We will periodically check partner's implementation to verify this is implemented as expected.

#Video signed urls
By default video urls are not returned in videos list response (search, top videos, category videos, etc.). However, we understand that in some cases user experience and efficiency requires having the video file available when fetching a list (usually when you want to give your user preview ability, on, say, hover). To get video with urls please provide the additional query parameter urls=true like so:

GET /videos?urls=true
The example above will return a list of videos with urls object on every video in the list.

But when getting a single video, urls always will be present.

The urls object has the following values:

{
 "mp4": "string",
 "mp4_preview": "string",
 "mp4_download": "string"
}
The values in the fields above are signed urls with jwt token. This token is without time expiration, but it is associated with your API_KEY. There’s no explicit limitations with that, but keep it in mind when hotlinking the videos!

The names are self-explanatory and also there is a note on mp4_download above.

We would only note that the mp4_preview url returns low resolution video files and helps us better understand when a video file was requested to actually be viewed rather than viewed as a thumbnail (e.g. videos grid).

COMING SOON: We will add two more links to the mp4 key so you’ll be able to fetch a specific resolution (low, medium, high) and decide which version fits your app and user experience best.

Collections endpoints
Collections are curated lists of videos that are constantly updated by our editorial team as new free stock videos are added to our videos library.

#List collections
This endpoint will return a paginated list of all collections in the system, which will enable users to explore the library (rather than searching).

GET /collections
#Parameters
Param	Description
page	Page number. Number. Default: 0
page_size	Number of videos per page. Number. Default: 20
#Response
{
  "page": 0,
  "pages": 1,
  "page_size": 20,
  "total": 7,
  "hits": [
    {
      "id": "kJibI2VztM",
      "title": "Paris Is For Lovers",
      "slug": "paris-is-for-lovers",
      "description": "Paris is always a good idea. The picturesque City of Lights can feel like a dream as it bursts with old world charm. Take a stroll along the Seine, stop for a flaky croissant, and spot the Eiffel Tower far off in the distance. ",
      "cover_video": "xj9ZIsytBA",
      "tags": [
        "paris",
        "paris cafe",
        "love"
      ],
      "videos": [
        "ypWCoOJp3g",
        "YrKN1E6hBJ",
        "uksv1hYaNq",
        "jXsV4e60Sc"
      ],
      "published": false,
      "meta_title": "Free Paris is for lovers Stock Video Footage - Royalty Free Video Download | Coverr",
      "meta_description": "Beautiful Paris is for lovers stock video footage for use on your website or any project (personal or commercial). Find high quality royalty free Paris is for lovers videos on Coverr | Coverr\n",
      "cover_image": "https://storage.coverr.co/t/02nCOQJbsWWluISAnlGtvelIKFD01XnLDf"
    }
  ]
}
#Get a collection
Once you have a collection you're interested in, you can fetch its details by accessing this endpoint using the collection id as key:

GET /collections/:id
#Parameters
Param	Description
id	Collection's id
#Response
{
  "id": "kJibI2VztM",
  "title": "Paris Is For Lovers",
  "slug": "paris-is-for-lovers",
  "description": "Paris is always a good idea. The picturesque City of Lights can feel like a dream as it bursts with old world charm. Take a stroll along the Seine, stop for a flaky croissant, and spot the Eiffel Tower far off in the distance. ",
  "cover_video": "xj9ZIsytBA",
  "tags": [
    "paris",
    "paris cafe",
    "love"
  ],
  "videos": [
    "ypWCoOJp3g",
    "YrKN1E6hBJ",
    "uksv1hYaNq",
    "jXsV4e60Sc"
  ],
  "published": false,
  "meta_title": "Free Paris is for lovers Stock Video Footage - Royalty Free Video Download | Coverr",
  "meta_description": "Beautiful Paris is for lovers stock video footage for use on your website or any project (personal or commercial). Find high quality royalty free Paris is for lovers videos on Coverr | Coverr\n",
  "cover_image": "https://storage.coverr.co/t/02nCOQJbsWWluISAnlGtvelIKFD01XnLDf"
}
#Get videos in a collection
In order to list all videos in a specific collection, just query this endpoint:

GET /collections/:id/videos
#Parameters
Param	Description
page	Page number. Number. Default: 1
page_size	Number of videos per page. Number. Default: 20
query	Search videos by query. String. Default: '', see a note below
urls	Add urls property in the response. Boolean. Default: false
#Response
{
  "page": 0,
  "pages": 50,
  "page_size": 20,
  "total": 3602,
  "hits": [
    {
      "id": "S1YbPl1NfI",
      "created_at": "2020-11-12T08:55:42.710Z",
      "updated_at": "2020-11-12T08:55:42.710Z",
      "title": "Cutting Wood Building Material With a Circular Electric Saw",
      "poster": "https://storage.coverr.co/p/QatsCWWAorI71sZ33DkHZREGWruZCHsg",
      "thumbnail": "https://storage.coverr.co/t/QatsCWWAorI71sZ33DkHZREGWruZCHsg",
      "description": "A close-up of the circular miter saw lowers and cuts off material",
      "is_vertical": false,
      "tags": [
        "circular saw",
        "miter saw",
        "saw"
      ],
      "downloads": 4,
      "views": 1,
      "published_at": "2020-11-15T09:27:18.578Z",
      "aspect_ratio": "16:9",
      "duration": 11.625,
      "max_height": 1152,
      "max_width": 2048,
      "urls": {
        "mp4": "https://storage.coverr.co/videos/QatsCWWAorI71sZ33DkHZREGWruZCHsg?token={token}",
        "mp4_preview": "https://storage.coverr.co/videos/QatsCWWAorI71sZ33DkHZREGWruZCHsg/preview?token={token}",
        "mp4_download": "https://storage.coverr.co/videos/QatsCWWAorI71sZ33DkHZREGWruZCHsg/download?token={token}&filename=Cutting Wood Building Material With a Circular Electric Saw"
      }
    }
  ]
}

Categories endpoints
Categories are system-wide "buckets" of videos that provide the basic structure to navigate all videos in the system.

Categories are updated periodically as new videos are added and tagged by our team.

#List categories
This endpoint will return a paginated list of all categories in the system. This is useful to explore the library (as opposed to searching).

GET /categories
#Parameters
Param	Description
page	Page number. Number. Default: 0
page_size	Number of videos per page. Number. Default: 20
#Response
{
  "page": 0,
  "pages": 2,
  "page_size": 20,
  "total": 25,
  "hits": [
    {
      "id": "8XEAeHkazU",
      "cover_video": "bJ7KveC5oR",
      "description": "Business and work videos are all about showing different work environments, from working at cafés to open-plan offices, working from home, developing, drawing, creating and more",
      "meta_description": "Business and work videos are all about showing different work environments, from working at cafés to open-plan offices, working from home, developing, drawing, creating and more | Coverr",
      "meta_title": "Free business and work stock video footage | Coverr",
      "name": "Business & Work",
      "slug": "business-work",
      "tags": [
        "business",
        "work",
        "office"
      ],
      "title": "Free business and work stock video footage",
      "cover_image": "https://storage.coverr.co/t/MrPvERg9QfKGZfck9kMpQKDia00j7DP8d",
      "order": 1
    }
  ]
}
#Get a category
Once you have a category you're interested in, you can fetch its details by accessing this endpoint using the collection slug or id as key:

GET /categories/:id
#Parameters
Param	Description
id	Category's id
#Response
{
  "id": "8XEAeHkazU",
  "cover_video": "bJ7KveC5oR",
  "description": "Business and work videos are all about showing different work environments, from working at cafés to open-plan offices, working from home, developing, drawing, creating and more",
  "meta_description": "Business and work videos are all about showing different work environments, from working at cafés to open-plan offices, working from home, developing, drawing, creating and more | Coverr",
  "meta_title": "Free business and work stock video footage | Coverr",
  "name": "Business & Work",
  "slug": "business-work",
  "tags": [
    "business",
    "work",
    "office"
  ],
  "title": "Free business and work stock video footage",
  "cover_image": "https://storage.coverr.co/t/MrPvERg9QfKGZfck9kMpQKDia00j7DP8d",
  "order": 1
}
#Get videos in a category
In order to list all videos in a specific category, just query this endpoint:

GET /categories/:id/videos
#Parameters
Param	Description
page	Page number. Number. Default: 1
page_size	Number of videos per page. Number. Default: 20
query	Search videos by query. String. Default: '', see a note below
urls	Add urls property in the response. Boolean. Default: false
#Response
{
  "page": 0,
  "pages": 50,
  "page_size": 20,
  "total": 3602,
  "hits": [
    {
      "id": "S1YbPl1NfI",
      "created_at": "2020-11-12T08:55:42.710Z",
      "updated_at": "2020-11-12T08:55:42.710Z",
      "title": "Cutting Wood Building Material With a Circular Electric Saw",
      "poster": "https://storage.coverr.co/p/QatsCWWAorI71sZ33DkHZREGWruZCHsg",
      "thumbnail": "https://storage.coverr.co/t/QatsCWWAorI71sZ33DkHZREGWruZCHsg",
      "description": "A close-up of the circular miter saw lowers and cuts off material",
      "is_vertical": false,
      "tags": [
        "circular saw",
        "miter saw",
        "saw"
      ],
      "downloads": 4,
      "views": 1,
      "published_at": "2020-11-15T09:27:18.578Z",
      "aspect_ratio": "16:9",
      "duration": 11.625,
      "max_height": 1152,
      "max_width": 2048,
      "urls": {
        "mp4": "https://storage.coverr.co/videos/QatsCWWAorI71sZ33DkHZREGWruZCHsg?token={token}",
        "mp4_preview": "https://storage.coverr.co/videos/QatsCWWAorI71sZ33DkHZREGWruZCHsg/preview?token={token}",
        "mp4_download": "https://storage.coverr.co/videos/QatsCWWAorI71sZ33DkHZREGWruZCHsg/download?token={token}&filename=Cutting Wood Building Material With a Circular Electric Saw"
      }
    }
  ]
}