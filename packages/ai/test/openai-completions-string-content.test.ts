import { describe, expect, it } from "vitest";
import { stream } from "../src/api/openai-completions.ts";
import type { FetchFunction, Message, Model } from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

function createSseResponse(events: unknown[]): Response {
	const body = `${events.map((event) => `data: ${JSON.stringify(event)}`).join("\r\n\r\n")}\r\n\r\ndata: [DONE]\r\n\r\n`;
	return new Response(body, { headers: { "content-type": "text/event-stream" } });
}

const terminalEvent = {
	id: "chatcmpl-test",
	object: "chat.completion.chunk",
	created: 0,
	model: "test-model",
	choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
};

/** The subset of the outgoing request body these tests assert on. */
interface RequestBody {
	store?: unknown;
	messages: Array<{ role: string; content: unknown }>;
}

function makeModel(compat?: Model<"openai-completions">["compat"]): Model<"openai-completions"> {
	return {
		id: "repro-model",
		name: "Repro Model",
		api: "openai-completions",
		provider: "repro-provider",
		reasoning: true,
		baseUrl: "https://example.invalid/v1",
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128000,
		maxTokens: 4096,
		input: ["text"],
		compat,
	};
}

const textMessage = (text: string): Message => ({
	role: "user",
	content: [{ type: "text", text }],
	timestamp: Date.now(),
});

/** Run one request and return the parsed body pi actually sent upstream. */
async function sentBody(model: Model<"openai-completions">, messages: Message[]): Promise<RequestBody> {
	let body: RequestBody | undefined;
	const fetch: FetchFunction = async (_input, init) => {
		body = JSON.parse(init?.body as string) as RequestBody;
		return createSseResponse([terminalEvent]);
	};
	const context = normalizeContext({ messages, tools: [] });
	await stream(model, context, { apiKey: "test", fetch }).result();
	if (!body) throw new Error("fetch was never called");
	return body;
}

describe("openai-completions content-parts array compatibility", () => {
	it("sends text-only user content as a content-parts array by default", async () => {
		const body = await sentBody(makeModel(), [textMessage("hello")]);

		expect(body.messages[0].content).toEqual([{ type: "text", text: "hello" }]);
	});

	it("sends text-only user content as a plain string when supportsContentArray is false", async () => {
		const body = await sentBody(makeModel({ supportsContentArray: false }), [textMessage("hello")]);

		expect(body.messages[0].content).toBe("hello");
	});

	it("keeps the content-parts array when supportsContentArray is explicitly true", async () => {
		const body = await sentBody(makeModel({ supportsContentArray: true }), [textMessage("hello")]);

		expect(body.messages[0].content).toEqual([{ type: "text", text: "hello" }]);
	});

	it("joins multiple text parts with newlines when supportsContentArray is false", async () => {
		const message: Message = {
			role: "user",
			content: [
				{ type: "text", text: "first" },
				{ type: "text", text: "second" },
			],
			timestamp: Date.now(),
		};

		const body = await sentBody(makeModel({ supportsContentArray: false }), [message]);

		expect(body.messages[0].content).toBe("first\nsecond");
	});

	it("omits the store parameter when supportsStore is false", async () => {
		const body = await sentBody(makeModel({ supportsStore: false }), [textMessage("hello")]);

		expect(body).not.toHaveProperty("store");
	});

	it("includes the store parameter by default", async () => {
		const body = await sentBody(makeModel(), [textMessage("hello")]);

		expect(body).toHaveProperty("store");
	});
});
