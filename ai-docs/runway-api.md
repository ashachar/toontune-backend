Setup
Runway offers the most cutting-edge generative video models. You can start using our models in your application with only a few quick steps.

Set up an organization
First, sign up for an account in the developer portal. After signing up, you’ll be presented with an option to create a new organization. An organization corresponds to your integration, and contains resources like API keys and configuration.

Create a key
Once you’ve created an organization, click to the API Keys tab. You’ll create a new key, giving it a name. Call it something descriptive, like “Matt Testing” so you can revoke the key later if needed.

The key will be presented to you only once: immediately copy it someplace safe, like a password manager. We’ll never return the key in plaintext again; if you lose the key, you’ll need to disable it and create a new one.

Add credits
Before you can start using the API, you’ll need to add credits to your organization. Credits are used to pay for the compute resources used by your models. Visit the billing tab in the developer portal to add credits to your organization. A minimum payment of $10 (at $0.01 per credit) is required to get started.

Start your integration
Now that you have an organization and a key, you can start using Runway’s API.

Using your API key
When you make requests to the API, you’ll need to include your API key in the headers. Our SDKs will automatically add your key if it’s specified in the RUNWAYML_API_SECRET environment variable. You can export this in your shell for test purposes like so:

macOS and Linux
Windows
Terminal window
export RUNWAYML_API_SECRET="key_123456789012345678901234567890"

In a production environment, do not hard-code your key like this. Instead, securely load your key into environment variables using a secret manager or similar tool.

Using the API
Before starting, make sure you have followed the instructions in the setup page.

Talking to the API
Generating Video
Generating Images
In this example, we’ll use the gen4_turbo model to generate a video from an image using the text prompt “Generate a video”. You’ll want to replace the promptImage with a URL of an image and a promptText with your own text prompt.

Node
Python
Just testing
First, you’ll want to install the Runway SDK. You can do this with npm:

Terminal window
npm install --save @runwayml/sdk

In your code, you can now import the SDK and start making requests:

import RunwayML, { TaskFailedError } from '@runwayml/sdk';

const client = new RunwayML();

// Create a new image-to-video task using the "gen4_turbo" model
try {
  const task = await client.imageToVideo
    .create({
      model: 'gen4_turbo',
      // Point this at your own image file
      promptImage: 'https://example.com/image.jpg',
      promptText: 'Generate a video',
      ratio: '1280:720',
      duration: 5,
    })
    .waitForTaskOutput();

  console.log('Task complete:', task);
} catch (error) {
  if (error instanceof TaskFailedError) {
    console.error('The video failed to generate.');
    console.error(error.taskDetails);
  } else {
    console.error(error);
  }
}

Uploading base64 encoded images as data URIs
You can also upload base64 encoded images (as a data URI) instead of pointing to an external URL. This can be useful if you’re working with a local image file and want to avoid an extra network round trip to upload the image.

To do this, simply pass the base64 encoded image string as a data URI in the promptImage field instead of a URL. For more information about file types and size limits, see the Inputs page.

Node
Python
import fs from 'node:fs';
import RunwayML, { TaskFailedError } from '@runwayml/sdk';

const client = new RunwayML();

// Read the image file into a Buffer. Replace `example.png` with your own image path.
const imageBuffer = fs.readFileSync('example.png');

// Convert to a data URI. We're using `image/png` here because the input is a PNG.
const dataUri = `data:image/png;base64,${imageBuffer.toString('base64')}`;

// Create a new image-to-video task using the "gen4_turbo" model
try {
  const imageToVideo = await client.imageToVideo
    .create({
      model: 'gen4_turbo',
      // Point this at your own image file
      promptImage: dataUri,
      promptText: 'Generate a video',
      ratio: '1280:720',
      duration: 5,
    })
    .waitForTaskOutput();

  console.log('Task complete:', task);
} catch (error) {
  if (error instanceof TaskFailedError) {
    console.error('The video failed to generate.');
    console.error(error.taskDetails);
  } else {
    console.error(error);
  }
}

Models
The API exposes the following models (1 credit = $0.01):

Model Name
Input	Output	
Pricing
Learn More
gen4_turbo
Image	Video	
5 credits / sec
Guide Reference
gen4_aleph
Video + Text/Image	Video	
15 credits / sec
Guide Reference
gen4_image
Text/Image (References)	Image	
5 credits / 720p image
8 credits / 1080p image
Guide Reference
gen4_image_turbo
Text+Image (References)	Image	
2 credits / image
(any resolution)
Guide Reference
gen3a_turbo
Image	Video	
5 credits / sec
Guide Reference
upscale_v1
Video	Video	
2 credits / sec
Reference
act_two
Image or Video	Video	
5 credits / sec
Reference
Model outputs
Gen-3 Alpha Turbo outputs

720p videos in 5s or 10s duration in the following resolutions
1280:768
768:1280
Gen-4 Turbo outputs

720p videos in 5s or 10s duration in the following resolutions
1280:720
720:1280
1104:832
832:1104
960:960
1584:672

Runway API SDKs
Available SDKs
We provide SDKs as convenient helpers for interacting with our API. These SDKs use best practices and offer type safety, which helps to avoid errors and make it easier to write code.

Node.js
https://www.npmjs.com/package/@runwayml/sdk

The Node.js SDK includes TypeScript bindings. It is compatible with Node 18 and up, and can be installed with npm, yarn, or pnpm.

Python
https://pypi.org/project/runwayml/

The Python SDK includes type annotations compatible with MyPy. It is compatible with Python 3.8 and up.

Generating content
You can create content using our API using the methods documented in the API reference. For instance, POST /v1/text_to_image accepts input and produces an image as output.

Each API endpoint for starting a generation is available as a member on the SDKs. Here is a mapping of the API endpoints to the SDK methods:

Node
Python
Operation	API endpoint	Node.js SDK method
Generate an image	POST /v1/text_to_image	client.textToImage.create
Generate a video	POST /v1/image_to_video	client.imageToVideo.create
Video upscale	POST /v1/video_upscale	client.videoUpscale.create
Character performance	POST /v1/character_performance	client.characterPerformance.create
Calling these methods will create a task. A task is a record of the generation operation. The response from the method will look like this:

{
  "id": "17f20503-6c24-4c16-946b-35dbbce2af2f"
}

The id field is the unique identifier for the task. You can use this ID to retrieve the task status and output from the GET /v1/tasks/{id} endpoint, which is available as the tasks.retrieve method on the SDKs.

Node
Python
import RunwayML from '@runwayml/sdk';
const client = new RunwayML();

const task = await client.tasks.retrieve('17f20503-6c24-4c16-946b-35dbbce2af2f');
console.log(task);

The response from the method will look like this:

{
  "id": "17f20503-6c24-4c16-946b-35dbbce2af2f",
  "status": "PENDING",
  "createdAt": "2024-06-27T19:49:32.334Z"
}

The API reference documents the statuses that a task can be in, along with the fields that are available for each status.

Tasks are processed asychronously. The tasks.retrieve method returns the current status of the task, which you can poll until the task has completed. The task will eventually transition to a SUCCEEDED, CANCELED, or FAILED status.

When polling, we recommend using an interval of 5 seconds or more. You should also add jitter, and handle non-200 responses with exponential backoff. Avoid using fixed interval polling (such as with JavaScript’s setInterval), since latency from the API can cause the polling to be too frequent.

Built-in task polling
As a convenience, all SDK methods that return a task include a helper method that polls for the task output. This reduces the amount of code you need to write to wait for a task to complete.

Node
Python
The waitForTaskOutput method is present on the unawaited response from the create methods on textToImage, imageToVideo, and videoUpscale.

// ✅ Call `waitForTaskOutput` on the unawaited response from `create`
const imageTask = await client.textToImage
  .create({
    model: 'gen4_image',
    promptText: 'A beautiful sunset over a calm ocean',
    ratio: '1360:768',
  })
  .waitForTaskOutput();

console.log(imageTask.output[0]); // Print the URL of the generated image

// ✅ Getting the task ID for bookkeeping purposes

// Notice: no `await` here.
const imageTask = client.textToImage.create({
  model: 'gen4_image',
  promptText: 'A beautiful sunset over a calm ocean',
  ratio: '1360:768',
});

// Await the output of `create` to get the task ID
const taskId = (await imageTask).id;
console.log(taskId); // The task ID can be stored in your database, for instance.

// Wait for the task to complete. It is safe to await `waitForTaskOutput`
// after the output of `create` was awaited above.
const completedTask = await imageTask.waitForTaskOutput();

console.log(completedTask.output[0]); // Print the URL of the generated image

// ❌ If you await the response from `create`, the result not have access to
// `waitForTaskOutput`.
const awaitedImageTask = await client.textToImage.create({
  model: 'gen4_image',
  promptText: 'A beautiful sunset over a calm ocean',
  ratio: '1360:768',
});
const taskOutput = await awaitedImageTask.waitForTaskOutput();

If the task fails (that is, its status becomes FAILED), a TaskFailedError will be thrown. You should handle this error appropriately.

import { TaskFailedError } from '@runwayml/sdk';
try {
  const imageTask = await client.textToImage
    .create({
      model: 'gen4_image',
      promptText: 'A beautiful sunset over a calm ocean',
      ratio: '1360:768',
    })
    .waitForTaskOutput();
} catch (error) {
  if (error instanceof TaskFailedError) {
    // `taskDetails` contains the output of the tasks.retrieve call.
    console.error('Task failed:', error.taskDetails);
  } else {
    throw error;
  }
}

The waitForTaskOutput method accepts an optional options parameter. This parameter can be used to specify a timeout and an AbortSignal.

const imageTask = await client.textToImage
  .create({
    model: 'gen4_image',
    promptText: 'A beautiful sunset over a calm ocean',
    ratio: '1360:768',
  })
  .waitForTaskOutput({
    // Wait up to 5 minutes for the task to complete
    timeout: 5 * 60 * 1000,
    // Abort the task if the request is cancelled
    abortSignal: myAbortSignal,
  });

By default, waitForTaskOutput will wait for ten minutes before timing out. Upon timeout, a TaskTimedOutError will be thrown. Pass null to timeout to wait indefinitely. Disabling the timeout is not recommended as it may cause your server to experience issues if your Runway API organization reaches its concurrency limit or if Runway experiences an outage.

It is recommended to use an AbortSignal to cancel polling if you are using waitForTaskOutput in the handler for an incoming request, such as on a web server. Here is an example of how to correctly integrate this into your application:

Express.js
Koa
Socket.io
const runway = new RunwayML();

app.post('/generate-image', (req, res) => {
  // Create an AbortController that triggers when the request is closed
  // unexpectedly.
  const abortController = new AbortController();
  req.on('close', () => {
    abortController.abort();
  });

  // 🚨 When performing a generation, be sure to add appropriate rate limiting
  // and other safeguards to prevent abuse.
  try {
    const imageTask = await runway.textToImage
      .create({
        model: 'gen4_image',
        promptText: req.body.prompt,
        ratio: '1360:768',
      })
      .waitForTaskOutput({ abortSignal: abortController.signal });

    res.send(imageTask.output[0]);
  } catch (error) {
    if (error instanceof TaskFailedError) {
      res.status(500).send('Task failed');
    } else {
      throw error;
    }
  }
});

Danger

Triggering the abortSignal passed to waitForTaskOutput or hitting the passed timeout will not cancel the task. Cancelling the task must be done by invoking the cancellation endpoint.

In addition to the methods that create new tasks, the tasks.retrieve method also returns a promise with a waitForTaskOutput method. This method is equivalent to the waitForTaskOutput method on the unawaited response from the create methods.

const task = await client.tasks
  .retrieve('17f20503-6c24-4c16-946b-35dbbce2af2f')
  .waitForTaskOutput();
console.log(task.output[0]);

This is useful if you’d like to create a task in one request and wait for its output in another request, or for handling the case where the client disconnected before the task completed.

Be aware that you must still add error handling for TaskFailedError and TaskTimeoutError when using this method.

Go-live checklist

Inputs
When starting tasks through the Runway API, you’ll often need to provide assets like images. Some restrictions exist for what you can provide.

Assets can be provided via URLs or Data URIs.

URLs
In all cases, URLs must meet some basic minimum requirements:

All URLs must be HTTPS.
URLs must reference a domain name, not an IP address.
The server should respond with valid Content-Type and Content-Length headers.
Redirects are not followed. If the URL returns a 3XX response code, the request is considered failed.
The length of any single URL should not exceed 2048 characters.
The file size of the image asset that the URL points to should not exceed 16MB.
Additionally, the server responding to the request must support HTTP HEAD requests.

Content-Type values
When specifying a URL, the Content-Type response header must be specified, and it must match the media type of your asset. File extensions in URLs are not considered. The Content-Types that are supported are listed below for the supported asset types.

Be aware that application/octet-stream and other generic values are explicitly not supported.

User agent
Runway will use a User-Agent header that starts with RunwayML API/ when making requests to your server. If you use a scraping-prevention tool or WAF, be sure to allowlist our user agent string prefix.

Data URIs (base64 encoded images)
A data URI allows you to pass the base64 encoded images as part of a request to our API, rather than passing a URL to the asset hosted on another server. This can reduce the complexity of your integration by eliminating an upload step.

Data URIs are supported anywhere URLs are expected. However, they come with some restrictions:

The length of the encoded data URI must be under 5MB (1024 × 1024 × 5 bytes). Keep in mind that base64-encoding your asset increases its size by about 33%: this means that you may not be able to use data URIs with assets larger than about 3.3MB. This limit supersedes type-specific file size limits.
The data URI must include an appropriate content type string. For instance, your data URI should start with something like data:image/jpg;base64,.
If a data URI is not base64 encoded, it may not be accepted.

Considerations
If you do not already have your asset stored in object storage, submitting your asset with a data URI can save you a step. Using a data URI may also help to reduce the latency of API calls.

However, the ~3MB limit may be too small for some assets, especially for video. If you cannot be sure that all assets are safely within the 5MB un-encoded size limit, you should upload assets to object storage instead. Uploaded assets (in other words, using a URL) have a limit of 16MB per image.

Type-specific requirements
Images
For fields that accept images, the asset referenced by the URL must use one of the following codecs, along with the corresponding Content-Type header:

Codec	Content-Type header
JPEG	image/jpg or image/jpeg
PNG	image/png
WebP	image/webp
All images are limited to 16MB.

Videos
For fields that accept videos, the asset referenced by the URL must use one of the following codecs, along with the corresponding Content-Type header:

Codec	Content-Type header
MP4	video/mp4
webm	video/webm
mov	video/quicktime or video/mov
Ogg	video/ogg
H.246	video/h264
All videos are limited to 16MB.

Aspect ratios and auto-cropping
Gen-4 Turbo and Act-Two support Landscape 1280:720 1584:672 1104:832, Portrait 720:1280 832:1104 and Square 960:960 outputs.

Gen-3 Alpha Turbo supports 1280:768 or 768:1280 outputs.

If your input asset is not exactly of the above listed ratios, the model will auto-crop your asset from the center to the aspect ratio parameter provided.

SDKs

Outputs
After a task succeeds, the GET /v1/tasks/:id endpoint will return a response like this:

{
  "id": "d2e3d1f4-1b3c-4b5c-8d46-1c1d7ee86892",
  "status": "SUCCEEDED",
  "createdAt": "2024-06-27T19:49:32.335Z",
  "output": [
    "https://dnznrvs05pmza.cloudfront.net/output.mp4?_jwt=..."
  ]
}

The output member will contain one or more URLs that link to the result of your generation.

It’s important to note that these URLs are ephemeral: they will expire within 24-48 hours of accessing the API. We expect you to download the data at this endpoint and save it to your own storage. Since these URLs will expire, do not expose them directly in your product.

