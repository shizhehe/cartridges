from cartridges.contexts.mcp.base import MCPContext
from cartridges.synthesizers.self_study_mcp import PromptSampler


class SlackMCPContext(MCPContext):

    class Config(MCPContext.Config):
        bot_token: str
        team_id: str
        
    def __init__(self, config: Config):
        command = "docker"
        args = [
            "run",
            "-i",
            "--rm",
            "-e", "SLACK_BOT_TOKEN",
            "-e", "SLACK_TEAM_ID",
            "mcp/slack"
        ]
        env = {
            "SLACK_BOT_TOKEN": config.bot_token,
            "SLACK_TEAM_ID": config.team_id
        }
        super().__init__(config, command=command, args=args, env=env)
    
    async def sample_subcontext(self):
        if self.session is None:
            raise RuntimeError("Not connected to an MCP server. You must call `connect_to_server` first.")
    
        out = await self.session.call_tool(
            name="slack_list_channels",
            arguments={
                "cursor": 0,
                "limit": 1,
            }
        )
        return str(out)

