export type ApiResponse<T> = {
  code: number;
  message: string;
  data: T;
};

export type WorkspacePage = {
  title: string;
  description: string;
};
