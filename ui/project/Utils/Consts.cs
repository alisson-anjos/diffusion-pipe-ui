namespace DiffusionPipeInterface.Utils
{
    public static class Consts
    {
        public const string CAPTIONER_SYSTEM_CONTEXT = @"You are a professional video analyst. Please provide an analysis of this video by covering each of these aspects in your answer. Use only one paragraph. DO NOT separate your answer into topics.";
        public const string CAPTIONER_PROMPT = @"1. **Main Content:**
    * What is the primary focus of the scene?
    * Who are the main characters visible?

2. **Object and Character Details:**
    * Don't refer to characters as 'individual', 'characters' and 'persons', instead always use their gender or refer to them with their gender.
    * Describe the appearance in detail
    * What notable objects are present?

3. **Actions and Movement:**
    * Describe ALL movements, no matter how subtle.
    * Specify the exact type of movement (walking, running, etc.).
    * Note the direction and speed of movements.

4. **Background Elements:**
    * Describe the setting and environment.
    * Note any environmental changes.

5. **Visual Style:**
    * Describe the lighting and color palette.
    * Note any special effects or visual treatments.
    * What is the overall style of the video? (e.g., realistic, animated, artistic, documentary)

6. **Camera Work:**
    * Describe EVERY camera angle change.
    * Note the distance from subjects (close-up, medium, wide shot).
    * Describe any camera movements (pan, tilt, zoom).

7. **Scene Transitions:**
    * How does each shot transition to the next?
    * Note any changes in perspective or viewing angle.

Please be extremely specific and detailed in your description. If you notice any movement or changes, describe them explicitly.";

    }
}
