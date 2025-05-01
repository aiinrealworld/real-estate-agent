from pydantic_ai import Agent, RunContext
from models import UserProfile
from typing import Optional, List

import logfire

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(
    send_to_logfire='if-token-present')

class RealEstateAgent(Agent[UserProfile, str]):
    def __init__(self, model: str = 'openai:gpt-4o', system_prompt: Optional[str] = None):
        if system_prompt is None:
            system_prompt = """
                You are a helpful real estate agent. Your goal is to gather information from the user to fill out their profile so you can find suitable properties for them.

                Your strategy is as follows:

                1. Examine the UserProfile object that is passed to you. This object contains the user's information. Check which fields are already filled.
                2. For each field that is NOT filled (i.e., is None or an empty list), ask the user a question to gather that information. Ask ONE question at a time. Be conversational.
                3. Immediately after the user provides the information, use the `update_profile` tool to update the UserProfile with the new information.
                4. Repeat steps 1-3 until all fields in the UserProfile are filled.
                5. Once all fields are filled, respond to the user "Thank you! I now have everything I need to find you the perfect properties."

                The UserProfile object has the following fields:

                - name: User's name
                - phone: User's phone number
                - location: Desired location
                - property_type: Type of property (e.g., house, apartment)
                - bedrooms: Number of bedrooms
                - bathrooms: Number of bathrooms
                - must_haves: List of must-have features
                - good_to_haves: List of good-to-have features

                Example:

                If the UserProfile object has name=None and phone=None, first ask: "Hello! What is your name?"
                After the user replies with their name, IMMEDIATELY use the `update_profile` tool.
                Then, see that phone=None and then ask: "Great! And what is your phone number?"

                Be polite and helpful. Respond naturally as a human real estate agent would.
            """
        super().__init__(model=model, deps_type=UserProfile, output_type=str, system_prompt=system_prompt, instrument=True)

    def _update_field(self, profile: UserProfile, field_name: str, new_value: any) -> bool:
        """Helper function to update a single field in the UserProfile."""
        if new_value is not None:
            setattr(profile, field_name, new_value)
            return True
        return False

    def _update_list_field(self, profile: UserProfile, field_name: str, new_items: Optional[List[str]]) -> bool:
         """Helper function to extend list fields (must_haves, good_to_haves)."""
         if new_items:
             current_list = getattr(profile, field_name)
             current_list.extend(new_items)
             return True
         return False

    @Agent.tool
    async def update_profile(self, ctx: RunContext[UserProfile], 
                             name: Optional[str] = None, 
                             phone: Optional[str] = None, 
                             location: Optional[str] = None, 
                             property_type: Optional[str] = None, 
                             bedrooms: Optional[int] = None, 
                             bathrooms: Optional[int] = None, 
                             must_haves: Optional[List[str]] = None, 
                             good_to_haves: Optional[List[str]] = None) -> str:
        """Updates the user profile with the provided information."""

        print(f"update_profile called with: name={name}, phone={phone}, ...")  # Log the parameters
        print(f"Current profile: {ctx.deps}")  # Log the current profile

        updated = False

        updated |= self._update_field(ctx.deps, 'name', name)
        updated |= self._update_field(ctx.deps, 'phone', phone)
        updated |= self._update_field(ctx.deps, 'location', location)
        updated |= self._update_field(ctx.deps, 'property_type', property_type)
        updated |= self._update_field(ctx.deps, 'bedrooms', bedrooms)
        updated |= self._update_field(ctx.deps, 'bathrooms', bathrooms)
        updated |= self._update_list_field(ctx.deps, 'must_haves', must_haves)
        updated |= self._update_list_field(ctx.deps, 'good_to_haves', good_to_haves)

        if updated:
             return "Updated the profile."
        else:
             return "No updates provided."