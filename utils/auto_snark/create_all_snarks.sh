#!/bin/bash

echo "🎙️ Creating all 6 cynical remarks for Do-Re-Mi scene..."
echo ""

# Create each snark with macOS say command
say -v Samantha "Oh good, another musical number. How original." -o temp.aiff
ffmpeg -y -i temp.aiff -acodec libmp3lame -ab 96k snark_1_musical.mp3 2>/dev/null
echo "✅ Created: snark_1_musical.mp3 (3.8s - Opening)"

say -v Alex "Yes, because A B C is such complex knowledge." -o temp.aiff  
ffmpeg -y -i temp.aiff -acodec libmp3lame -ab 96k snark_2_abc.mp3 2>/dev/null
echo "✅ Created: snark_2_abc.mp3 (16.8s - After ABC line)"

say -v Samantha "We get it. You can repeat three syllables." -o temp.aiff
ffmpeg -y -i temp.aiff -acodec libmp3lame -ab 96k snark_3_repeat.mp3 2>/dev/null
echo "✅ Created: snark_3_repeat.mp3 (25.5s - After repetition)"

say -v Daniel "Easier? This is your idea of teaching?" -o temp.aiff
ffmpeg -y -i temp.aiff -acodec libmp3lame -ab 96k snark_4_teaching.mp3 2>/dev/null
echo "✅ Created: snark_4_teaching.mp3 (41.8s - After 'make it easier')"

say -v Alex "Revolutionary. A deer is... a deer." -o temp.aiff
ffmpeg -y -i temp.aiff -acodec libmp3lame -ab 96k snark_5_deer.mp3 2>/dev/null
echo "✅ Created: snark_5_deer.mp3 (48.0s - After deer definition)"

say -v Samantha "Me, the narcissism is showing." -o temp.aiff
ffmpeg -y -i temp.aiff -acodec libmp3lame -ab 96k snark_6_narcissism.mp3 2>/dev/null
echo "✅ Created: snark_6_narcissism.mp3 (52.5s - After 'Mi, a name I call myself')"

rm temp.aiff

echo ""
echo "✨ All 6 cynical remarks created!"
echo ""
echo "📊 File sizes:"
ls -lh snark_*.mp3

echo ""
echo "🎧 Play sample with: afplay snark_1_musical.mp3"